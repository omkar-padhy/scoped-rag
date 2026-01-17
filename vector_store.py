from __future__ import annotations

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# BM25 for hybrid search
from rank_bm25 import BM25Okapi

from model import get_embeddings
from llm import get_llm, query_with_fallback
from config import DB_PATH

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ENABLE_QUERY_EXPANSION = False  # Disabled - was causing bad retrieval
ENABLE_HYBRID_SEARCH = True     # BM25 + Vector search
ENABLE_RERANKING = True         # Multi-signal reranking
ENABLE_GEMINI_RERANK = True     # Use Gemini for semantic relevance scoring
RERANK_TOP_K = 6                # Final number of docs after reranking


# ============================================================================
# Gemini Semantic Reranker
# ============================================================================

class GeminiReranker:
    """Use Gemini to score document relevance semantically"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                self.client = client
                self.model_name = 'gemini-2.0-flash'
                logger.info("✅ Gemini reranker initialized")
            except Exception as e:
                logger.warning(f"❌ Gemini reranker failed: {e}")
                self.client = None
        else:
            self.client = None
        
        self._initialized = True
    
    def score_relevance(self, query: str, documents: list[Document]) -> list[tuple[float, Document]]:
        """Score documents for relevance to query using Gemini"""
        if not self.client or not documents:
            return [(0.5, doc) for doc in documents]
        
        try:
            # Create compact document summaries for scoring
            doc_summaries = []
            for i, doc in enumerate(documents[:10]):  # Limit to 10 for speed
                content = doc.page_content[:300].replace('\n', ' ')
                doc_summaries.append(f"[{i}] {content}")
            
            prompt = f"""Score each document's relevance to the query on a scale of 0.0 to 1.0.
Return ONLY a JSON array of scores like [0.8, 0.6, 0.9, ...] in document order.

Query: {query}

Documents:
{chr(10).join(doc_summaries)}

JSON scores:"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            text = response.text.strip()
            
            # Parse JSON array
            import json
            import re
            # Extract JSON array from response
            match = re.search(r'\[[\d\.,\s]+\]', text)
            if match:
                scores = json.loads(match.group())
                # Ensure we have enough scores
                while len(scores) < len(documents):
                    scores.append(0.5)
                return [(scores[i] if i < len(scores) else 0.5, doc) for i, doc in enumerate(documents)]
            
        except Exception as e:
            logger.debug(f"Gemini scoring failed: {e}")
        
        return [(0.5, doc) for doc in documents]


def get_gemini_reranker() -> GeminiReranker:
    """Get singleton Gemini reranker"""
    return GeminiReranker()


# ============================================================================
# BM25 Hybrid Search
# ============================================================================

class BM25Index:
    """Simple BM25 index for keyword search"""
    
    def __init__(self):
        self.documents: list[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []
    
    def add_documents(self, docs: list[Document]):
        """Add documents to BM25 index"""
        self.documents.extend(docs)
        self._rebuild_index()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing"""
        return text.lower().split()
    
    def _rebuild_index(self):
        """Rebuild BM25 index from documents"""
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self.documents
        ]
        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)
    
    def search(self, query: str, k: int = 5, access_level: int = 5) -> list[Document]:
        """Search documents using BM25"""
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices with access level filtering
        scored_docs = []
        for i, score in enumerate(scores):
            doc = self.documents[i]
            doc_level = doc.metadata.get("access_level", 5)
            if doc_level <= access_level:
                scored_docs.append((score, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:k]]
    
    def clear(self):
        """Clear the index"""
        self.documents = []
        self.bm25 = None
        self._tokenized_corpus = []


# Global BM25 index (rebuilt when store is loaded)
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Get or create BM25 index"""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def rebuild_bm25_index(store: Chroma):
    """Rebuild BM25 index from Chroma store"""
    global _bm25_index
    _bm25_index = BM25Index()
    
    try:
        # Get all documents from Chroma
        results = store.get(include=["documents", "metadatas"])
        if results and results["documents"]:
            docs = []
            for i, content in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                docs.append(Document(page_content=content, metadata=metadata))
            _bm25_index.add_documents(docs)
            logger.info(f"✅ BM25 index rebuilt with {len(docs)} documents")
    except Exception as e:
        logger.warning(f"Failed to rebuild BM25 index: {e}")


# ============================================================================
# Query Classification
# ============================================================================

class QueryClassifier:
    """Classify query intent for better retrieval strategy"""
    
    # Order matters! More specific patterns should come first
    QUERY_TYPES = [
        ("eligibility", ["eligible", "eligibility", "criteria", "requirement", "qualify", "who can apply"]),
        ("contact", ["who to contact", "contact", "phone", "email", "call", "reach", "officer"]),
        ("date", ["when", "date", "deadline", "last date", "duration"]),  # Move date before process
        ("process", ["how to", "steps", "procedure", "process", "apply", "application"]),
        ("list", ["list", "all", "types", "categories", "enumerate"]),
        ("amount", ["how much", "amount", "funding", "grant", "fellowship amount", "salary"]),
        ("definition", ["what is", "define", "meaning", "explain", "describe"]),
    ]
    
    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query into a type for retrieval optimization"""
        query_lower = query.lower()
        for qtype, keywords in cls.QUERY_TYPES:
            if any(kw in query_lower for kw in keywords):
                return qtype
        return "general"


# ============================================================================
# Evidence Detection for Reranking
# ============================================================================

def compute_evidence_score(query_type: str, content: str) -> float:
    """
    Score based on presence of extractable/answerable content.
    Returns 0.0 to 1.0
    """
    import re
    score = 0.0
    content_lower = content.lower()
    
    if query_type == "contact":
        # Look for ACTUAL contact data (not headers)
        # Phone numbers (Indian format)
        phones = re.findall(r'011-\d{7,8}', content)
        score += len(phones) * 0.25
        
        # Email patterns (obfuscated)
        emails = re.findall(r'\w+\[dot\]\w+\[at\]', content_lower)
        score += len(emails) * 0.25
        
        # Named contacts (Dr./Mr./Ms./Sh. followed by names)
        names = re.findall(r'(?:Dr\.|Mr\.|Ms\.|Sh\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', content)
        score += len(names) * 0.15
        
        # Complete contact blocks (name + phone + email together)
        if phones and emails and names:
            score += 0.5  # Big bonus for complete info
        
        # PENALTY: Table headers without actual data
        if re.search(r'Name of Officer.*Phone.*Email', content, re.IGNORECASE):
            if not phones and not emails:
                score -= 0.5  # Heavy penalty
    
    elif query_type == "definition":
        # Definitional patterns
        if re.search(r'(?:is|are|means|refers to|defined as|is a|is an)', content_lower):
            score += 0.3
        # First paragraph often has definitions
        if re.search(r'^[A-Z].*(?:is|are)\s+(?:a|an|the)', content):
            score += 0.2
    
    elif query_type == "process":
        # Step indicators
        steps = re.findall(r'(?:step\s*\d|^\d+\.|•|➢)', content_lower, re.MULTILINE)
        score += len(steps) * 0.1
        # Process keywords
        if re.search(r'(?:apply|submit|process|procedure|following)', content_lower):
            score += 0.2
    
    elif query_type == "eligibility":
        # Eligibility patterns
        if re.search(r'(?:eligible|criteria|must|should|requirement|qualification)', content_lower):
            score += 0.3
        # Age/experience patterns
        if re.search(r'(?:age|years|experience|degree|qualification)', content_lower):
            score += 0.2
    
    elif query_type == "date":
        # Date patterns
        dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', content)
        score += len(dates) * 0.2
        if re.search(r'(?:deadline|last date|due date|before|by)', content_lower):
            score += 0.2
    
    elif query_type == "amount":
        # Currency/amount patterns
        amounts = re.findall(r'(?:Rs\.?|INR|₹)\s*[\d,]+|[\d,]+\s*(?:lakh|crore|per month|per annum)', content)
        score += len(amounts) * 0.2
    
    return min(score, 1.0)


# ============================================================================
# Reranking with Multi-Signal Scoring
# ============================================================================

def rerank_documents(query: str, docs: list[Document], top_k: int = 6) -> list[Document]:
    """
    Rerank documents using multi-signal scoring:
    1. Lexical (keyword/phrase matching)
    2. Structure (chunk type matches query type)
    3. Evidence (contains actual answerable content)
    """
    if not ENABLE_RERANKING or len(docs) <= top_k:
        return docs[:top_k]
    
    try:
        import re
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        query_type = QueryClassifier.classify(query)
        
        logger.debug(f"Query type: {query_type}")
        
        # Extract important phrases (2-3 word combinations)
        words = query_lower.split()
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        scored_docs = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            content = doc.page_content
            
            # === Signal 1: Lexical Score (0-40 points) ===
            lexical_score = 0
            
            # Phrase matching (higher weight)
            for phrase in phrases:
                if phrase in content_lower:
                    lexical_score += 8
            
            # Individual term matching
            for term in query_terms:
                if len(term) > 2 and term in content_lower:
                    lexical_score += 1
            
            # Boost by filename match in chunk_id - CRITICAL for topic relevance
            chunk_id = doc.metadata.get("chunk_id", "").lower()
            filename_matches = sum(1 for term in query_terms if len(term) > 2 and term in chunk_id)
            lexical_score += filename_matches * 10  # Strong boost for topic match
            
            # Extra boost for acronyms in filename (TIF, IYBF, etc.)
            acronyms = [t.upper() for t in query_terms if len(t) >= 3 and t.upper() == t]
            if not acronyms:  # Check if any term is all caps or could be acronym
                acronyms = [t.upper() for t in query_terms if len(t) >= 2 and len(t) <= 5]
            for acr in acronyms:
                if acr.lower() in chunk_id:
                    lexical_score += 15  # Big bonus for acronym match in filename
            
            # Boost by heading context
            if doc.metadata.get("heading_context"):
                heading = doc.metadata["heading_context"].lower()
                if any(term in heading for term in query_terms if len(term) > 2):
                    lexical_score += 3
            
            lexical_score = min(lexical_score, 50)  # Increased cap
            
            # === Signal 2: Structure Score (0-20 points) ===
            structure_score = 0
            chunk_type = doc.metadata.get("chunk_type", "text")
            
            # Match chunk type to query type
            if query_type == "contact" and chunk_type in ["table", "contact"]:
                structure_score += 15
            elif query_type == "process" and chunk_type in ["list", "procedure"]:
                structure_score += 15
            elif query_type == "definition" and chunk_type in ["paragraph", "intro"]:
                structure_score += 10
            
            # === Signal 3: Evidence Score (0-50 points) ===
            evidence_score = compute_evidence_score(query_type, content) * 50
            
            # === Combined Score ===
            total_score = lexical_score + structure_score + evidence_score
            
            scored_docs.append((total_score, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # === Signal 4: Gemini Semantic Score (optional) ===
        if ENABLE_GEMINI_RERANK:
            try:
                # Get top candidates for Gemini scoring
                top_candidates = [doc for _, doc in scored_docs[:12]]
                reranker = get_gemini_reranker()
                gemini_scored = reranker.score_relevance(query, top_candidates)
                
                # Combine: 60% original score + 40% Gemini score
                final_scored = []
                original_scores = {id(doc): score for score, doc in scored_docs[:12]}
                max_original = max(original_scores.values()) if original_scores else 1
                
                for gemini_score, doc in gemini_scored:
                    orig_score = original_scores.get(id(doc), 0)
                    # Normalize original to 0-1 range
                    norm_orig = orig_score / max_original if max_original > 0 else 0
                    combined = (norm_orig * 0.6) + (gemini_score * 0.4)
                    final_scored.append((combined, doc))
                
                final_scored.sort(key=lambda x: x[0], reverse=True)
                reranked = [doc for _, doc in final_scored[:top_k]]
                logger.info(f"Gemini reranked {len(docs)} docs → top {len(reranked)}")
                return reranked
            except Exception as e:
                logger.debug(f"Gemini reranking failed, using fallback: {e}")
        
        # Log top scores for debugging
        for score, doc in scored_docs[:3]:
            chunk_id = doc.metadata.get("chunk_id", "?")
            logger.debug(f"  Score {score:.1f}: {chunk_id[:50]}")
        
        reranked = [doc for _, doc in scored_docs[:top_k]]
        logger.info(f"Reranked {len(docs)} docs → top {len(reranked)} (query_type: {query_type})")
        return reranked
        
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return docs[:top_k]


# ============================================================================
# Hybrid Retrieval (Vector + BM25)
# ============================================================================

def hybrid_retrieve(
    store: Chroma,
    query: str,
    user_level: int = 5,
    vector_k: int = 8,
    bm25_k: int = 4
) -> list[Document]:
    """
    Hybrid retrieval combining vector search (semantic) and BM25 (keyword).
    Returns deduplicated, merged results.
    Also fetches sibling chunks for context.
    """
    import re
    all_docs = []
    seen_ids = set()
    
    # 1. Vector search (semantic)
    search_kwargs = {
        "k": vector_k,
        "fetch_k": vector_k * 3,
        "lambda_mult": 0.7,
        "filter": {"access_level": {"$lte": user_level}}
    }
    
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    vector_docs = retriever.invoke(query)
    for doc in vector_docs:
        doc_id = doc.metadata.get("chunk_id", id(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            all_docs.append(doc)
    
    # 2. BM25 search (keyword) if enabled
    if ENABLE_HYBRID_SEARCH:
        bm25_index = get_bm25_index()
        bm25_docs = bm25_index.search(query, k=bm25_k, access_level=user_level)
        
        for doc in bm25_docs:
            doc_id = doc.metadata.get("chunk_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
        
        logger.debug(f"Hybrid search: {len(vector_docs)} vector + {len(bm25_docs)} BM25 → {len(all_docs)} unique")
    
    # 3. Sibling chunk retrieval: Fetch adjacent chunks from same files
    # This helps when contact data is in a separate chunk from the main content
    sibling_docs = fetch_sibling_chunks(store, all_docs, user_level)
    for doc in sibling_docs:
        doc_id = doc.metadata.get("chunk_id", id(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            all_docs.append(doc)
    
    if sibling_docs:
        logger.debug(f"Added {len(sibling_docs)} sibling chunks")
    
    return all_docs


def fetch_sibling_chunks(store: Chroma, docs: list[Document], user_level: int) -> list[Document]:
    """
    Fetch adjacent chunks from the same files as retrieved docs.
    Helps capture contact data that might be in nearby chunks.
    Also fetches contact block chunks (p99:cX) from the same file.
    """
    import re
    sibling_docs = []
    seen_files = set()
    seen_chunk_ids = {doc.metadata.get("chunk_id", "") for doc in docs}
    
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "")
        if not chunk_id:
            continue
        
        # Parse chunk_id format: "filename:pX:cY"
        match = re.match(r"(.+):p(\d+):c(\d+)$", chunk_id)
        if not match:
            continue
        
        filename = match.group(1)
        page = int(match.group(2))
        chunk_num = int(match.group(3))
        
        # Skip if we've already processed this file
        if filename in seen_files:
            continue
        seen_files.add(filename)
        
        # Fetch adjacent chunks (c-1, c+1, c+2)
        for offset in [-1, 1, 2]:
            sibling_num = chunk_num + offset
            if sibling_num < 0:
                continue
            
            sibling_chunk_id = f"{filename}:p{page}:c{sibling_num}"
            
            # Skip if already in results
            if sibling_chunk_id in seen_chunk_ids:
                continue
            
            try:
                # Use metadata filter to find the sibling chunk
                results = store.get(
                    where={"chunk_id": sibling_chunk_id},
                    include=["documents", "metadatas"]
                )
                
                if results and results["documents"] and len(results["documents"]) > 0:
                    content = results["documents"][0]
                    meta = results["metadatas"][0] if results["metadatas"] else {}
                    
                    # Check access level
                    if meta.get("access_level", 5) <= user_level:
                        sibling_doc = Document(
                            page_content=content,
                            metadata=meta
                        )
                        sibling_docs.append(sibling_doc)
                        seen_chunk_ids.add(sibling_chunk_id)
                        logger.debug(f"Fetched sibling: {sibling_chunk_id}")
            except Exception as e:
                # Sibling doesn't exist or error, skip
                logger.debug(f"Could not fetch sibling {sibling_chunk_id}: {e}")
        
        # Also fetch contact block chunks (p99:cX) from this file
        # Contact blocks are stored with page=99 as a marker
        for contact_idx in range(10):  # Up to 10 contact blocks
            contact_chunk_id = f"{filename}:p99:c{contact_idx}"
            
            if contact_chunk_id in seen_chunk_ids:
                continue
            
            try:
                results = store.get(
                    where={"chunk_id": contact_chunk_id},
                    include=["documents", "metadatas"]
                )
                
                if results and results["documents"] and len(results["documents"]) > 0:
                    content = results["documents"][0]
                    meta = results["metadatas"][0] if results["metadatas"] else {}
                    
                    if meta.get("access_level", 5) <= user_level:
                        contact_doc = Document(
                            page_content=content,
                            metadata=meta
                        )
                        sibling_docs.append(contact_doc)
                        seen_chunk_ids.add(contact_chunk_id)
                        logger.debug(f"Fetched contact block: {contact_chunk_id}")
            except Exception:
                break  # No more contact blocks
    
    return sibling_docs


def expand_query(question: str) -> list[str]:
    """
    Generate query variations to improve retrieval coverage.
    Returns original query + 2-3 reformulations.
    """
    if not ENABLE_QUERY_EXPANSION:
        return [question]
    
    expansion_prompt = f"""Generate 3 alternative search queries for this question. 
Keep them short and focused on key terms. Return ONLY the queries, one per line.

Original: {question}

Alternatives:"""
    
    try:
        # Use query_with_fallback for automatic rate limit handling
        response = query_with_fallback(expansion_prompt)
        variations = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Return original + up to 3 variations
        result = [question] + variations[:3]
        logger.debug(f"Query expanded: {question} → {len(result)} variations")
        return result
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [question]  # Fallback to original only


def create_vector_store(
    docs: list[Document],
) -> Chroma:
    embedding = get_embeddings()
    # Chroma handles persistence automatically with valid path
    store = Chroma(
        collection_name="scoped_rag",
        embedding_function=embedding,
        persist_directory=str(DB_PATH),
    )
    if docs:
        store.add_documents(docs)
        print(f"Created/Updated Chroma store at {DB_PATH} with {len(docs)} docs")
    return store


def add_documents_to_store(store: Chroma, docs: list[Document]):
    if not docs:
        return
    store.add_documents(docs)
    print(f"Added {len(docs)} documents to store")
    
    # Update BM25 index
    if ENABLE_HYBRID_SEARCH:
        bm25_index = get_bm25_index()
        bm25_index.add_documents(docs)


def delete_documents_by_filename(store: Chroma, filename: str):
    """Delete all documents associated with a filename from the store."""
    try:
        # We query by 'file_name' metadata. 
        # Note: If existing index doesn't have 'file_name', this might fail or do nothing.
        # We rely on new indexing to put 'file_name'.
        # For 'source' (which was absolute path in some cases), it is harder.
        # But we will use 'file_name' going forward.
        
        # Chroma 'get' with where filter
        results = store.get(where={"file_name": filename})
        if results and results["ids"]:
            store.delete(ids=results["ids"])
            print(f"Deleted {len(results['ids'])} chunks for {filename}")
        else:
            # Try 'source' as fallback for legacy (if source == filename)
            results = store.get(where={"source": filename})
            if results and results["ids"]:
                store.delete(ids=results["ids"])
                print(f"Deleted {len(results['ids'])} chunks for {filename} (via source)")
            else:
                 # Try 'source' as absolute path (if we can constructing it)
                 # But we don't know the absolute path easily here without DATA_PATH
                 pass

    except Exception as e:
        print(f"Error deleting documents for {filename}: {e}")


def save_vector_store(store: Chroma, path: str = "") -> None:
    # Chroma validates persistence on initialization/update usually
    pass


def load_vector_store(path: str = "") -> Chroma:
    embedding = get_embeddings()
    store = Chroma(
        collection_name="scoped_rag",
        embedding_function=embedding,
        persist_directory=DB_PATH,
    )
    print(f"Loaded Chroma store from {DB_PATH}")
    
    # Rebuild BM25 index for hybrid search
    if ENABLE_HYBRID_SEARCH:
        rebuild_bm25_index(store)
    
    return store


# Improved prompt template for better RAG responses
RAG_PROMPT_TEMPLATE = """You are a precise document extraction assistant. Extract information ONLY from the documents below.

=== DOCUMENTS START ===
{context}
=== DOCUMENTS END ===

{chat_history}QUESTION: {question}

EXTRACTION RULES:
1. ONLY use information that appears EXACTLY in the documents above
2. For contacts: Extract EXACT names, phone numbers, and emails as written
3. Convert email formats: [dot] → . and [at] → @
4. If information is not in the documents, say: "Not found in the provided documents"
5. NEVER invent or fabricate any names, numbers, or emails

EXAMPLES:
- Document has "Dr. Sanjay Kumar Mishra" → Write "Dr. Sanjay Kumar Mishra" (exact)
- Document has "011-24361035" → Write "011-24361035" (exact)
- Document has "sanjaykr[dot]mishra[at]nic[dot]in" → Write "sanjaykr.mishra@nic.in"
- Document has table headers but no data → Say "Contact table found but details not visible"

Format your answer clearly with bullet points or numbered lists when appropriate.

ANSWER:"""


def format_chat_history(chat_history: list[dict], max_turns: int = 3) -> str:
    """
    Format recent chat history for context.
    Only includes last N turns to avoid overwhelming the LLM.
    
    Args:
        chat_history: List of {"role": "user"|"assistant", "content": str}
        max_turns: Maximum number of Q&A pairs to include (default 3)
    
    Returns:
        Formatted string of recent conversation
    """
    if not chat_history:
        return ""
    
    # Filter to only user/assistant messages, get last N*2 messages
    relevant = [m for m in chat_history if m.get("role") in ("user", "assistant")]
    recent = relevant[-(max_turns * 2):]
    
    if not recent:
        return ""
    
    lines = ["RECENT CONVERSATION:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages
        content = msg.get("content", "")[:300]
        if len(msg.get("content", "")) > 300:
            content += "..."
        lines.append(f"{role}: {content}")
    
    lines.append("")  # Empty line before current question
    return "\n".join(lines) + "\n"


def create_rag_chain(store: Chroma):
    llm = get_llm()
    # MMR search for diversity, fetch more candidates then select best 8
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 25,
            "lambda_mult": 0.7  # Balance relevance vs diversity
        }
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain


def query_vector_store(store: Chroma, question: str, user_level: int = 5) -> str:
    """Query with hybrid search (vector + BM25), query expansion, and reranking"""
    # Expand query for better coverage
    queries = expand_query(question)
    
    # Collect docs from all query variations
    all_docs = []
    seen_ids = set()
    
    for q in queries:
        # Hybrid retrieval (vector + BM25)
        docs = hybrid_retrieve(store, q, user_level, vector_k=4, bm25_k=2)
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    
    # Rerank and limit to top docs
    all_docs = rerank_documents(question, all_docs, top_k=RERANK_TOP_K)

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    context = format_docs(all_docs)
    
    logger.info(f"📚 Retrieved {len(all_docs)} docs for query")
    
    # Build and invoke chain
    llm, model_name = get_llm()
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        return response.content
    return str(response)


def query_with_sources(
    store: Chroma, 
    question: str, 
    user_level: int = 5,
    chat_history: list[dict] = None
) -> dict[str, object]:
    """
    Query with hybrid search, source tracking, and optional chat history context.
    
    Args:
        store: Chroma vector store
        question: User's question
        user_level: Access level (1-5)
        chat_history: Optional list of previous messages for context
    """
    import re
    
    # Classify query for optimized retrieval
    query_type = QueryClassifier.classify(question)
    logger.info(f"🔍 Query type: {query_type}")
    
    # Direct retrieval without query expansion - increase counts for better coverage
    all_docs = hybrid_retrieve(store, question, user_level, vector_k=12, bm25_k=6)
    
    # Rerank and limit to top docs
    all_docs = rerank_documents(question, all_docs, top_k=RERANK_TOP_K)
    
    logger.info(f"📚 Retrieved {len(all_docs)} docs for query")

    llm, model_name = get_llm()
    context = "\n\n---\n\n".join(doc.page_content for doc in all_docs)
    
    # Clean problematic characters that can confuse LLM APIs
    context = context.replace('\x00', '')  # Null bytes from OCR
    context = context.replace('\xa0', ' ')  # Non-breaking spaces
    
    # Format chat history (last 3 turns only)
    history_text = format_chat_history(chat_history or [], max_turns=3)

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context, 
        chat_history=history_text,
        question=question
    )

    response = llm.invoke(prompt)
    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = str(response)
    
    # === Clean up LLM response ===
    # Remove repeated header garbage that sometimes appears
    answer = clean_llm_response(answer)
    
    # === Output Validation: Detect Hallucinated Contacts ===
    hallucination_warning = None
    if query_type == "contact":
        answer, hallucination_warning = validate_contact_response(answer, context)

    # === Compute Confidence Score ===
    confidence = compute_confidence_score(question, all_docs, query_type)

    result = {
        "answer": answer,
        "sources": [doc.metadata.get("chunk_id", "N/A") for doc in all_docs],
        "context": context,
        "model": model_name,
        "query_type": query_type,
        "confidence": confidence,
    }
    
    if hallucination_warning:
        result["warning"] = hallucination_warning
        logger.warning(f"⚠️ {hallucination_warning}")
    
    return result


def compute_confidence_score(query: str, docs: list[Document], query_type: str) -> dict:
    """
    Compute confidence score for the retrieval based on multiple signals.
    Returns dict with overall score (0-1) and breakdown.
    """
    import re
    
    if not docs:
        return {"score": 0.0, "level": "none", "signals": {}}
    
    signals = {}
    
    # === Signal 1: Document count coverage ===
    doc_count = len(docs)
    if doc_count >= 6:
        signals["coverage"] = 1.0
    elif doc_count >= 3:
        signals["coverage"] = 0.7
    elif doc_count >= 1:
        signals["coverage"] = 0.4
    else:
        signals["coverage"] = 0.0
    
    # === Signal 2: Query term match rate ===
    query_terms = set(query.lower().split())
    # Remove stop words
    stop_words = {"what", "is", "the", "how", "who", "for", "to", "in", "of", "a", "an", "and", "or", "can", "i", "get"}
    query_terms = query_terms - stop_words
    
    if query_terms:
        match_count = 0
        all_content = " ".join(doc.page_content.lower() for doc in docs)
        for term in query_terms:
            if term in all_content:
                match_count += 1
        signals["term_match"] = match_count / len(query_terms)
    else:
        signals["term_match"] = 0.5  # Neutral if no meaningful terms
    
    # === Signal 3: Evidence density (for contact queries) ===
    if query_type == "contact":
        phone_pattern = r'\d{3}[-.\s]?\d{4,8}'
        email_pattern = r'[\w\.\-]+@[\w\.\-]+'
        
        all_content = " ".join(doc.page_content for doc in docs)
        phones = re.findall(phone_pattern, all_content)
        emails = re.findall(email_pattern, all_content)
        
        evidence_count = len(set(phones)) + len(set(emails))
        if evidence_count >= 3:
            signals["evidence_density"] = 1.0
        elif evidence_count >= 1:
            signals["evidence_density"] = 0.6
        else:
            signals["evidence_density"] = 0.2
    else:
        signals["evidence_density"] = 0.5  # Neutral for non-contact
    
    # === Signal 4: Chunk type relevance ===
    relevant_types = 0
    for doc in docs:
        chunk_type = doc.metadata.get("chunk_type", "text")
        if query_type == "contact" and chunk_type in ["table", "contact"]:
            relevant_types += 1
        elif query_type == "process" and chunk_type in ["list", "procedure"]:
            relevant_types += 1
        elif query_type in ["definition", "eligibility"] and chunk_type in ["paragraph", "text"]:
            relevant_types += 1
    
    signals["type_relevance"] = min(relevant_types / max(len(docs), 1), 1.0)
    
    # === Weighted average ===
    weights = {
        "coverage": 0.2,
        "term_match": 0.35,
        "evidence_density": 0.25,
        "type_relevance": 0.2
    }
    
    overall = sum(signals[k] * weights[k] for k in weights)
    
    # Determine confidence level
    if overall >= 0.75:
        level = "high"
    elif overall >= 0.5:
        level = "medium"
    elif overall >= 0.25:
        level = "low"
    else:
        level = "very_low"
    
    return {
        "score": round(overall, 2),
        "level": level,
        "signals": {k: round(v, 2) for k, v in signals.items()}
    }


def clean_llm_response(response: str) -> str:
    """
    Clean up LLM response by removing garbage patterns.
    - Removes repeated header blocks
    - Removes model artifacts (like header_end tags)
    - Strips leading/trailing whitespace
    """
    import re
    
    # Remove common model artifacts
    response = re.sub(r'<\|header_end\|>', '', response)
    response = re.sub(r'assistant<\|[^>]+\|>', '', response)
    response = re.sub(r'\bassistant\b', '', response, flags=re.IGNORECASE)
    
    # Remove repeated "India" pattern
    response = re.sub(r'(?:India)+', '', response, flags=re.IGNORECASE)
    
    # Remove repeated department header blocks
    # Pattern: multiple occurrences of "DBT\nDepartment of\nBiotechnology..." 
    header_pattern = r'(?:\s*(?:DBT|Department of|Ministry of|Government of|Education|Biotechnology|Science & Technology)\s*[\n]*){3,}'
    response = re.sub(header_pattern, '', response, flags=re.IGNORECASE)
    
    # Remove very long repeated lines (like logos/navigation repeated)
    lines = response.split('\n')
    cleaned_lines = []
    seen_lines = set()
    for line in lines:
        line_stripped = line.strip()
        # Skip empty lines at the beginning
        if not cleaned_lines and not line_stripped:
            continue
        # Skip exact duplicate lines (but allow empty lines within content)
        if line_stripped and line_stripped in seen_lines and len(line_stripped) < 50:
            continue
        cleaned_lines.append(line)
        if line_stripped:
            seen_lines.add(line_stripped)
    
    response = '\n'.join(cleaned_lines).strip()
    
    return response


def validate_contact_response(response: str, context: str) -> tuple[str, str | None]:
    """
    Validate that contact information in response exists in context.
    Returns (response, warning_message or None)
    """
    import re
    
    # Extract phone numbers from response and context
    # Indian phone patterns: 011-XXXXXXXX, +91-XX-XXXX-XXXX, etc.
    phone_pattern = r'\d{3}[-.\s]?\d{4,8}'
    
    response_phones = set(re.findall(phone_pattern, response))
    context_phones = set(re.findall(phone_pattern, context))
    
    # Normalize phone numbers (remove separators)
    def normalize_phone(p):
        return re.sub(r'[-.\s]', '', p)
    
    response_phones_norm = {normalize_phone(p) for p in response_phones}
    context_phones_norm = {normalize_phone(p) for p in context_phones}
    
    # Find hallucinated phones (check if any context phone is a substring)
    hallucinated_phones = set()
    for rp in response_phones_norm:
        # A phone is hallucinated if no context phone contains it (or it contains context phone)
        found = False
        for cp in context_phones_norm:
            if rp in cp or cp in rp or rp[-8:] == cp[-8:]:  # Last 8 digits match
                found = True
                break
        if not found:
            hallucinated_phones.add(rp)
    
    # Extract emails from response and context
    email_pattern = r'[\w\.\-]+@[\w\.\-]+'
    response_emails = set(re.findall(email_pattern, response.lower()))
    
    # Context may have obfuscated emails like name[dot]surname[at]domain[dot]tld
    context_lower = context.lower()
    context_emails = set(re.findall(email_pattern, context_lower))
    
    # Parse various obfuscated email formats
    # Pattern 1: user[dot]name[at]domain[dot]tld[dot]suffix
    obfuscated1 = re.findall(r'([\w]+(?:\[dot\][\w]+)*)\[at\]([\w]+(?:\[dot\][\w]+)*)', context_lower)
    for user_part, domain_part in obfuscated1:
        user = user_part.replace('[dot]', '.')
        domain = domain_part.replace('[dot]', '.')
        # Add both with and without .in suffix (for truncated emails)
        context_emails.add(f"{user}@{domain}")
        if not domain.endswith('.in') and not domain.endswith('.com'):
            context_emails.add(f"{user}@{domain}.in")
    
    # Check hallucinated emails - more lenient matching
    hallucinated_emails = set()
    for re_mail in response_emails:
        re_user = re_mail.split('@')[0]
        found = False
        for ce_mail in context_emails:
            ce_user = ce_mail.split('@')[0]
            # Match if username parts are same (ignore minor domain differences)
            if re_user == ce_user or re_mail == ce_mail:
                found = True
                break
        if not found:
            hallucinated_emails.add(re_mail)
    
    warning = None
    if hallucinated_phones or hallucinated_emails:
        details = []
        if hallucinated_phones:
            details.append(f"phones: {list(hallucinated_phones)[:3]}")
        if hallucinated_emails:
            details.append(f"emails: {list(hallucinated_emails)[:3]}")
        warning = f"Potential hallucination detected: {', '.join(details)}"
    
    return response, warning
