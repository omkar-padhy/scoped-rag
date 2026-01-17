# RAG System Improvement Plan

> **Goal**: Transform the current RAG system into a production-grade document intelligence platform with reliable retrieval, accurate responses, and zero hallucinations.

## ✅ Implementation Status (Updated: Jan 2025)

### Phase 1: Quick Wins - ✅ COMPLETED

- [x] Query classification (`QueryClassifier` in vector_store.py)
- [x] Evidence-based reranking (`compute_evidence_score`)
- [x] Output validation (`validate_contact_response`)
- [x] chunk_type metadata support
- [x] Fixed null byte bug in OCR (causing hallucinations)
- [x] Fixed .env loading for API keys

### Phase 2: Chunking Overhaul - ✅ COMPLETED

- [x] Table-aware chunking in ocr.py
- [x] Contact block extraction (`_detect_contact_block`, `_extract_contact_blocks`)
- [x] Contact blocks stored with `page=99` marker
- [x] Sibling chunk retrieval (`fetch_sibling_chunks`)
- [x] Full re-index completed (4087 chunks, 18 contact blocks)

### Phase 3: Advanced Retrieval - ✅ COMPLETED

- [x] Multi-signal reranking (lexical + structure + evidence)
- [x] Gemini cross-encoder integration (`GeminiReranker` class)
- [x] Hybrid search (Vector + BM25)

### Phase 4: Production Hardening - ✅ COMPLETED

- [x] Confidence scoring (`compute_confidence_score`)
- [x] Response cleaning (`clean_llm_response`)
- [x] Hallucination detection with warnings

---

## Table of Contents

1. [Current System Analysis](#1-current-system-analysis)
2. [Chunking Strategy Improvements](#2-chunking-strategy-improvements)
3. [Retrieval Strategy Improvements](#3-retrieval-strategy-improvements)
4. [Reranking Strategy Improvements](#4-reranking-strategy-improvements)
5. [Prompt Engineering Improvements](#5-prompt-engineering-improvements)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Current System Analysis

### 1.1 Current Architecture

```
Documents → OCR/Parsing → Chunking → Embeddings → ChromaDB
                                          ↓
User Query → Hybrid Search (Vector + BM25) → Reranking → LLM → Response
```

### 1.2 Current Issues Identified

| Component     | Issue                                                   | Impact                                          |
| ------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **Chunking**  | Fixed 512 char chunks break tables, contacts, lists     | Critical data split across chunks               |
| **Chunking**  | No special handling for structured data (tables, forms) | Table headers separated from data               |
| **Retrieval** | Basic term matching in reranking                        | Wrong chunks ranked higher                      |
| **Retrieval** | No semantic understanding of query intent               | Contact queries don't prioritize contact chunks |
| **LLM**       | Hallucination despite temperature=0                     | Fake names/emails generated                     |
| **LLM**       | Context too long causes confusion                       | LLM ignores relevant chunks                     |

### 1.3 Root Cause: TIF Contact Issue

The TIF Programme contact hallucination happened because:

1. **Chunk 1**: Contains table header ("Name of Officer", "Phone No.", "Email")
2. **Chunk 2**: Contains actual contact data (Dr. Sanjay Kumar Mishra, etc.)
3. **Only Chunk 1 was retrieved** → LLM saw headers without data → Hallucinated

---

## 2. Chunking Strategy Improvements

### 2.1 Problem: Naive Fixed-Size Chunking

Current approach splits text at arbitrary boundaries:

```python
# Current (BAD)
RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
```

**Result**: Tables get split mid-row, contact blocks get fragmented.

### 2.2 Solution: Semantic-Aware Chunking

#### A. Document Structure Detection

```python
class SemanticChunker:
    """Chunk based on document structure, not character count"""

    STRUCTURAL_PATTERNS = {
        "table": r"^\s*\|.*\|.*\|",  # Markdown tables
        "contact_block": r"(Name|Phone|Email|Contact).*:.*",
        "numbered_list": r"^\s*\d+\.\s+",
        "section_header": r"^#{1,3}\s+|\n[A-Z][A-Z\s]{5,50}\n",
    }

    def chunk(self, text: str, metadata: dict) -> list[Document]:
        # 1. Detect structural elements
        # 2. Keep related content together
        # 3. Add context from parent sections
```

#### B. Table-Aware Chunking

**Rule**: Never split tables. Always include header row with data rows.

```python
def preserve_tables(text: str) -> list[str]:
    """Extract tables as single chunks"""
    table_pattern = r"(\|[^\n]+\|\n)+"  # Markdown table
    tables = re.findall(table_pattern, text)

    # Each table becomes ONE chunk (even if >1000 chars)
    return tables
```

#### C. Contact Block Detection

**Rule**: Keep contact information together.

```python
CONTACT_PATTERNS = [
    r"(?:Name|Contact).*?(?:Email|Phone|Mobile).*?[\w\.\-]+@[\w\.\-]+",  # Full contact
    r"(?:Dr\.|Mr\.|Ms\.|Sh\.)\s+[A-Z][a-z]+\s+[A-Z].*?\d{3,}.*?@",     # Name + Phone + Email
]

def extract_contact_blocks(text: str) -> list[str]:
    """Extract complete contact entries as atomic chunks"""
    pass
```

#### D. Hierarchical Chunking with Parent Context

```python
@dataclass
class HierarchicalChunk:
    content: str
    parent_heading: str  # "Contacts Concerned Officer for more Information"
    grandparent_heading: str  # "TATA Innovation Fellowship"
    chunk_type: str  # "table", "paragraph", "list", "contact"
```

### 2.3 Recommended Chunk Sizes by Type

| Content Type       | Min Size | Max Size  | Overlap | Notes               |
| ------------------ | -------- | --------- | ------- | ------------------- |
| **Tables**         | -        | Unlimited | 0       | Never split         |
| **Contact Blocks** | -        | 800       | 0       | Keep atomic         |
| **Paragraphs**     | 200      | 600       | 100     | Standard text       |
| **Lists**          | -        | 1000      | 50      | Keep items together |
| **Code/Forms**     | -        | 1200      | 0       | Preserve structure  |

### 2.4 Implementation: SmartChunker Class

```python
class SmartChunker:
    def __init__(self):
        self.table_extractor = TableExtractor()
        self.contact_extractor = ContactExtractor()
        self.section_splitter = SectionSplitter()

    def chunk_document(self, content: ExtractedContent) -> list[Document]:
        chunks = []

        # Step 1: Extract atomic elements (tables, contacts)
        tables = self.table_extractor.extract(content.text_content)
        contacts = self.contact_extractor.extract(content.text_content)

        # Step 2: Remove extracted elements from main text
        remaining_text = self._remove_extracted(
            content.text_content, tables + contacts
        )

        # Step 3: Section-aware splitting for remaining text
        sections = self.section_splitter.split(remaining_text)

        # Step 4: Standard chunking within sections
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)

        # Step 5: Add atomic elements as separate chunks
        chunks.extend(self._create_table_chunks(tables))
        chunks.extend(self._create_contact_chunks(contacts))

        return chunks
```

---

## 3. Retrieval Strategy Improvements

### 3.1 Current Retrieval Pipeline

```
Query → [Vector Search] + [BM25 Search] → Merge → Rerank → Top-K
```

**Issues**:

- No query understanding
- No document type awareness
- No structural matching

### 3.2 Query Understanding Layer

#### A. Query Classification

```python
class QueryClassifier:
    QUERY_TYPES = {
        "contact": ["who", "contact", "phone", "email", "call", "reach"],
        "definition": ["what is", "define", "meaning", "explain"],
        "process": ["how to", "steps", "procedure", "process"],
        "list": ["list", "all", "types", "categories"],
        "comparison": ["difference", "compare", "vs", "versus"],
        "factual": ["when", "where", "how many", "date", "location"],
    }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        for qtype, keywords in self.QUERY_TYPES.items():
            if any(kw in query_lower for kw in keywords):
                return qtype
        return "general"
```

#### B. Query-Aware Retrieval

```python
def retrieve_with_intent(query: str, store: Chroma) -> list[Document]:
    query_type = QueryClassifier().classify(query)

    if query_type == "contact":
        # Prioritize chunks with phone/email patterns
        return retrieve_contact_chunks(query, store)
    elif query_type == "definition":
        # Prioritize intro paragraphs and glossary
        return retrieve_definition_chunks(query, store)
    elif query_type == "process":
        # Prioritize numbered lists and procedures
        return retrieve_process_chunks(query, store)
    else:
        return hybrid_retrieve(query, store)
```

### 3.3 Multi-Stage Retrieval

```
Stage 1: Coarse Retrieval (k=50)
    ↓
Stage 2: Type-Based Filtering
    ↓
Stage 3: Semantic Reranking (k=20)
    ↓
Stage 4: Final Selection (k=6)
```

#### Implementation

```python
def multi_stage_retrieve(
    query: str,
    store: Chroma,
    query_type: str
) -> list[Document]:

    # Stage 1: Broad retrieval
    candidates = hybrid_retrieve(query, store, k=50)

    # Stage 2: Type-based filtering
    if query_type == "contact":
        candidates = [d for d in candidates
                      if has_contact_data(d.page_content)]

    # Stage 3: Cross-encoder reranking
    scored = cross_encoder_rerank(query, candidates)

    # Stage 4: Diversity selection (avoid redundant chunks)
    final = select_diverse_top_k(scored, k=6)

    return final
```

### 3.4 Chunk Type Metadata for Filtering

Add during indexing:

```python
chunk.metadata["chunk_type"] = detect_chunk_type(content)
# Values: "table", "contact", "paragraph", "list", "heading", "code"
```

Use during retrieval:

```python
if query_type == "contact":
    filter = {"chunk_type": {"$in": ["contact", "table"]}}
    candidates = store.similarity_search(query, filter=filter, k=20)
```

### 3.5 Parent-Child Retrieval

**Problem**: Small relevant chunk found, but context needed.

**Solution**: Store parent-child relationships.

```python
@dataclass
class ChunkWithContext:
    chunk: Document
    parent_chunk: Optional[Document]  # Larger context
    sibling_chunks: list[Document]    # Same section
```

When a chunk is retrieved, optionally fetch its parent:

```python
def retrieve_with_context(query: str, store: Chroma) -> list[Document]:
    chunks = standard_retrieve(query, store)

    expanded = []
    for chunk in chunks:
        expanded.append(chunk)

        # Fetch parent if chunk is small
        if len(chunk.page_content) < 300:
            parent_id = chunk.metadata.get("parent_chunk_id")
            if parent_id:
                parent = store.get(ids=[parent_id])
                expanded.append(parent)

    return deduplicate(expanded)
```

---

## 4. Reranking Strategy Improvements

### 4.1 Current Reranking Issues

```python
# Current: Simple keyword matching
score = phrase_matches * 10 + term_matches * 1
```

**Problems**:

- No semantic understanding
- "Contact" in query doesn't boost chunks with actual phone numbers
- No cross-encoder for deep relevance scoring

### 4.2 Solution: Multi-Signal Reranking

#### A. Signal Types

| Signal              | Weight | Description                   |
| ------------------- | ------ | ----------------------------- |
| **Semantic Score**  | 0.4    | Cross-encoder similarity      |
| **Lexical Score**   | 0.2    | BM25/TF-IDF                   |
| **Structure Score** | 0.2    | Chunk type matches query type |
| **Evidence Score**  | 0.2    | Contains extractable answers  |

#### B. Evidence Detection

```python
def compute_evidence_score(query: str, chunk: str) -> float:
    """Score based on presence of answerable content"""
    score = 0.0

    query_type = classify_query(query)

    if query_type == "contact":
        # Look for actual contact data (not headers)
        phone_pattern = r"\d{3}[-.\s]?\d{4,8}"
        email_pattern = r"[\w\.\-]+@[\w\.\-]+"

        phones = re.findall(phone_pattern, chunk)
        emails = re.findall(email_pattern, chunk)

        score += len(phones) * 0.3
        score += len(emails) * 0.3

        # Bonus for complete contact (name + phone + email nearby)
        if phones and emails:
            score += 0.4

    elif query_type == "definition":
        # Look for definitional patterns
        if re.search(r"(?:is|are|means|refers to|defined as)", chunk.lower()):
            score += 0.5

    return min(score, 1.0)
```

#### C. Cross-Encoder Reranking (Optional)

```python
from sentence_transformers import CrossEncoder

class SemanticReranker:
    def __init__(self):
        # Lightweight model for fast reranking
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs]
```

### 4.3 Combined Reranking Function

```python
def advanced_rerank(
    query: str,
    docs: list[Document],
    top_k: int = 6
) -> list[Document]:

    query_type = classify_query(query)

    scored_docs = []
    for doc in docs:
        # Signal 1: Lexical (current BM25-style)
        lexical = compute_lexical_score(query, doc.page_content)

        # Signal 2: Structure match
        structure = compute_structure_score(query_type, doc.metadata)

        # Signal 3: Evidence presence
        evidence = compute_evidence_score(query, doc.page_content)

        # Signal 4: Semantic (optional, requires cross-encoder)
        # semantic = cross_encoder.score(query, doc.page_content)

        # Combined score
        final_score = (
            lexical * 0.3 +
            structure * 0.3 +
            evidence * 0.4
        )

        scored_docs.append((final_score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
```

---

## 5. Prompt Engineering Improvements

### 5.1 Current Prompt Issues

- LLM ignores actual data in documents
- Generates plausible-sounding fake information
- Doesn't distinguish between headers and data

### 5.2 Solution: Structured Extraction Prompt

```python
RAG_PROMPT_TEMPLATE = """You are a precise document extraction assistant.

TASK: Extract information from the DOCUMENTS to answer the question.

=== DOCUMENTS START ===
{context}
=== DOCUMENTS END ===

QUESTION: {question}

INSTRUCTIONS:
1. ONLY use information explicitly written in the DOCUMENTS
2. If the answer is not in the documents, respond: "Not found in documents"
3. For contact queries:
   - Find lines containing actual names (e.g., "Dr. John Smith")
   - Find actual phone numbers (e.g., "011-24361035")
   - Convert email formats: [dot] → . and [at] → @
4. Quote the exact text when possible
5. Never invent names, numbers, or emails

EXTRACTION RULES:
- ✓ CORRECT: "Dr. Sanjay Kumar Mishra - 011-24361035" (from document)
- ✗ WRONG: "Dr. S. K. Chaudhary - 011-2751-4404" (invented)

ANSWER:"""
```

### 5.3 Few-Shot Examples (For Contact Queries)

```python
CONTACT_FEW_SHOT = """
Example 1:
DOCUMENT: "Contacts: 1. Dr. Manoj Singh - 011-24363726 - manoj@dbt.nic.in"
QUESTION: "Who to contact?"
ANSWER: **Dr. Manoj Singh** - Phone: 011-24363726 - Email: manoj@dbt.nic.in

Example 2:
DOCUMENT: "Contact the office for more details. Phone: [number not provided]"
QUESTION: "What is the phone number?"
ANSWER: Phone number not provided in the document.
"""
```

### 5.4 Output Validation

Post-process LLM output to detect hallucinations:

```python
def validate_contacts_in_response(response: str, context: str) -> str:
    """Remove contacts from response that don't exist in context"""

    # Extract all contact patterns from response
    response_phones = re.findall(r"\d{3}[-.\s]?\d{4,8}", response)
    response_emails = re.findall(r"[\w\.\-]+@[\w\.\-]+", response)

    # Extract patterns from context
    context_phones = set(re.findall(r"\d{3}[-.\s]?\d{4,8}", context))
    context_emails = set(re.findall(r"[\w\.\-]+@[\w\.\-]+", context))

    # Check for hallucinated contacts
    hallucinated_phones = [p for p in response_phones if p not in context_phones]
    hallucinated_emails = [e for e in response_emails if e not in context_emails]

    if hallucinated_phones or hallucinated_emails:
        logger.warning(f"Detected hallucinated contacts: {hallucinated_phones}, {hallucinated_emails}")
        # Option 1: Remove hallucinated content
        # Option 2: Return error message
        # Option 3: Re-query with stricter prompt

    return response
```

---

## 6. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

| Task                     | File              | Priority | Impact                       |
| ------------------------ | ----------------- | -------- | ---------------------------- |
| Add chunk_type metadata  | `ocr.py`          | High     | Enables type-based filtering |
| Query classification     | `vector_store.py` | High     | Improves retrieval relevance |
| Evidence-based reranking | `vector_store.py` | High     | Fixes contact hallucination  |
| Output validation        | `vector_store.py` | Medium   | Catches hallucinations       |

### Phase 2: Chunking Overhaul (3-5 days)

| Task                       | File      | Priority | Impact                    |
| -------------------------- | --------- | -------- | ------------------------- |
| Table-aware chunking       | `ocr.py`  | Critical | Keeps tables intact       |
| Contact block extraction   | `ocr.py`  | Critical | Keeps contacts intact     |
| Parent-child relationships | `ocr.py`  | Medium   | Enables context expansion |
| Re-index all documents     | `main.py` | Required | Apply new chunking        |

### Phase 3: Advanced Retrieval (5-7 days)

| Task                      | File              | Priority | Impact                 |
| ------------------------- | ----------------- | -------- | ---------------------- |
| Multi-stage retrieval     | `vector_store.py` | High     | Better precision       |
| Cross-encoder reranking   | `vector_store.py` | Medium   | Semantic understanding |
| Query expansion (careful) | `vector_store.py` | Low      | Better recall          |

### Phase 4: Production Hardening (3-5 days)

| Task                 | File              | Priority | Impact               |
| -------------------- | ----------------- | -------- | -------------------- |
| Response validation  | `server.py`       | High     | Catch hallucinations |
| Confidence scoring   | `vector_store.py` | Medium   | User trust           |
| Caching improvements | `server.py`       | Medium   | Performance          |
| Logging & monitoring | All               | Medium   | Debugging            |

---

## 7. Testing Strategy

### 7.1 Test Cases for Contact Queries

```python
TEST_CASES = [
    {
        "query": "Who to contact for TIF Programme?",
        "expected_names": ["Dr. Sanjay Kumar Mishra", "Dr. Manoj Singh Rohilla"],
        "expected_phones": ["011-24361035", "011-24363726"],
        "forbidden": ["Dr. S. K. Chaudhary", "Dr. R. K. Singh"]  # Hallucinated
    },
    # Add more test cases...
]

def run_retrieval_tests():
    for case in TEST_CASES:
        response = query_with_sources(store, case["query"])

        # Check expected content is present
        for name in case["expected_names"]:
            assert name in response["answer"], f"Missing: {name}"

        # Check forbidden (hallucinated) content is absent
        for forbidden in case["forbidden"]:
            assert forbidden not in response["answer"], f"Hallucinated: {forbidden}"
```

### 7.2 Metrics to Track

| Metric                 | Target | How to Measure          |
| ---------------------- | ------ | ----------------------- |
| **Retrieval Recall**   | >90%   | Relevant chunk in top-6 |
| **Hallucination Rate** | 0%     | Contacts not in source  |
| **Response Accuracy**  | >95%   | Manual evaluation       |
| **Latency**            | <3s    | P95 response time       |

---

## 8. Quick Implementation: Contact Query Fix

### Immediate Fix (Apply Now)

```python
# In vector_store.py - rerank_documents function

def rerank_documents(query: str, docs: list[Document], top_k: int = 6):
    query_lower = query.lower()
    is_contact_query = any(w in query_lower for w in
                          ['contact', 'phone', 'email', 'call', 'reach', 'who'])

    scored_docs = []
    for doc in docs:
        content = doc.page_content
        score = 0

        # Base scoring (existing)
        # ...

        # CRITICAL: For contact queries, heavily boost chunks with ACTUAL data
        if is_contact_query:
            # Phone numbers (Indian format: 011-XXXXXXXX)
            phones = re.findall(r'011-\d{7,8}', content)
            score += len(phones) * 20  # High boost

            # Email patterns
            emails = re.findall(r'[\w\.]+\[(?:dot|at)\]', content.lower())
            score += len(emails) * 20

            # Named contacts (Dr./Mr./Ms. followed by names)
            names = re.findall(r'(?:Dr\.|Mr\.|Ms\.|Sh\.)\s+[A-Z][a-z]+\s+[A-Z]', content)
            score += len(names) * 15

            # PENALTY: Chunks with table headers but no data
            if re.search(r'Name of Officer.*Phone.*Email', content, re.IGNORECASE):
                if not phones and not emails:
                    score -= 30  # Heavy penalty for header-only chunks

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
```

---

## Summary

The key improvements needed are:

1. **Chunking**: Keep tables and contacts as atomic units
2. **Retrieval**: Query-type awareness + multi-stage filtering
3. **Reranking**: Evidence-based scoring (actual data > headers)
4. **Prompts**: Stricter extraction rules + validation
5. **Testing**: Automated tests for common failure cases

Start with the quick win in Section 8 to immediately fix the contact hallucination issue.
