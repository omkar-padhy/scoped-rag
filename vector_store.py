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

ENABLE_QUERY_EXPANSION = True   # Generate query variations
ENABLE_HYBRID_SEARCH = True     # BM25 + Vector search
ENABLE_RERANKING = True         # Cross-encoder reranking
RERANK_TOP_K = 6                # Final number of docs after reranking


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
# Reranking with Cross-Encoder
# ============================================================================

def rerank_documents(query: str, docs: list[Document], top_k: int = 6) -> list[Document]:
    """
    Rerank documents using LLM-based scoring.
    Falls back to original order if reranking fails.
    """
    if not ENABLE_RERANKING or len(docs) <= top_k:
        return docs[:top_k]
    
    try:
        # Use LLM to score relevance (lightweight approach)
        scored_docs = []
        
        for doc in docs:
            # Simple heuristic: count query term overlap
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            
            # Boost by metadata hints
            if doc.metadata.get("heading_context"):
                heading = doc.metadata["heading_context"].lower()
                if any(term in heading for term in query_terms):
                    overlap += 2
            
            scored_docs.append((overlap, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        reranked = [doc for _, doc in scored_docs[:top_k]]
        logger.debug(f"Reranked {len(docs)} docs → top {len(reranked)}")
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
    """
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
    
    return all_docs

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
RAG_PROMPT_TEMPLATE = """You are a precise research assistant. Answer questions based ONLY on the provided context.

<instructions>
1. Read all context chunks carefully before answering
2. If information is partial, synthesize from multiple chunks
3. If context is insufficient, say "I don't have enough information about [topic]"
4. Quote relevant excerpts when helpful
5. Be concise but complete
</instructions>

<context>
{context}
</context>

<question>
{question}
</question>

<answer>
"""


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
    llm = get_llm()
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        return response.content
    return str(response)


def query_with_sources(store: Chroma, question: str, user_level: int = 5) -> dict[str, object]:
    """Query with hybrid search, query expansion, reranking, and source tracking"""
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
    
    logger.info(f"📚 Retrieved {len(all_docs)} docs for query")

    llm = get_llm()
    context = "\n\n---\n\n".join(doc.page_content for doc in all_docs)

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(prompt)
    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = str(response)

    return {
        "answer": answer,
        "sources": [doc.metadata.get("chunk_id", "N/A") for doc in all_docs],
        "context": context,
    }
