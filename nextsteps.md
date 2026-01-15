# RAG System Improvement Plan

## Executive Summary

After analyzing the full codebase, I've identified several areas that are likely causing poor RAG answer quality. The issues span embedding models, retrieval configuration, prompting strategy, and chunking logic.

---

## 🔴 Critical Issues (High Impact)

### 1. **Weak LLM Model** Done

- **Current:** `llama3.2:3b-instruct-q4_K_M` (3B parameters, Q4 quantized)
- **Problem:** 3B models struggle with complex reasoning and context synthesis
- **Fix:** Upgrade to `llama3.2:8b-instruct-q4_K_M` or `qwen2.5:7b-instruct-q4_K_M`
- **File:** `config.py` line 13

```python
# Before
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"

# After
LLM_MODEL = "llama3.2:8b-instruct-q4_K_M"  # or qwen2.5:7b-instruct
```

---

### 2. **Poor Retrieval Configuration (k=3-4 is too low)** Done

- **Current:** Only fetching 3-4 chunks
- **Problem:** Important context is being missed, especially for multi-topic questions
- **Fix:** Increase `k` and add **Reranking**
- **File:** `vector_store.py` lines 78, 108, 152

```python
# Before
search_kwargs = {"k": 4}

# After - Retrieve more, then rerank
search_kwargs = {"k": 10}  # Get more candidates

# Add reranking (new logic needed)
# Use cross-encoder or Cohere rerank to select top 4-5
```

---

### 3. **Missing Hybrid Search (Semantic + Keyword)** Done

- **Current:** Pure semantic search only
- **Problem:** Semantic search can miss exact terms (names, codes, technical terms)
- **Fix:** Add BM25/keyword search alongside vector search
- **File:** `vector_store.py`

```python
# Proposed: Use LangChain's EnsembleRetriever
from langchain.retrievers import EnsembleRetriever, BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=5)
vector = store.as_retriever(search_kwargs={"k": 5})
hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.3, 0.7])
```

---

### 4. **Prompts Are Too Generic** Done

- **Current:** Simple "use only context" instruction
- **Problem:** No guidance on answer format, reasoning, or handling ambiguity
- **File:** `vector_store.py` lines 83-93, 117-127, 164-175

**Improved Prompt Template:**

```python
prompt = ChatPromptTemplate.from_template("""
You are a precise research assistant. Answer questions based ONLY on the provided context.

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
""")
```

---

## 🟡 Medium Priority Issues

### 5. **Embedding Model Could Be Stronger** Done

- **Current:** `nomic-embed-text:137m-v1.5-fp16` (137M params)
- **Problem:** Small embedding models have weaker semantic understanding
- **Fix Options:**
  - Local: `mxbai-embed-large:335m-v1-fp16` (better quality)
  - API: `voyage-3-lite` or `text-embedding-3-small` (if API available)
- **File:** `config.py` line 14

---

### 6. **Chunk Size/Overlap May Not Be Optimal** Done

- **Current:** 1000 chars, 200 overlap
- **Problem:** Could be too large for detailed retrieval
- **Test Values:**
  - Try `chunk_size=512, overlap=128` for denser retrieval
  - Or `chunk_size=1500, overlap=300` if documents are long-form
- **Files:** `ocr.py` lines 743-744, `process_documents()` defaults

---

### 7. **No Query Expansion/Reformulation** Done

- **Problem:** Single query may not match document vocabulary
- **Fix:** Add HyDE (Hypothetical Document Embedding) or query expansion

```python
def expand_query(query: str, llm) -> list[str]:
    """Generate multiple query variations"""
    prompt = f"Generate 3 variations of this search query: {query}"
    variations = llm.invoke(prompt).split("\n")
    return [query] + variations[:3]
```

---

### 8. **Semantic Headers Add Noise** Done

- **Current:** Every chunk has prepended metadata header
- **Problem:** Can confuse retrieval and bloat token count
- **File:** `ocr.py` `_add_semantic_header()` method
- **Fix:** Store metadata separately, don't prepend to content

---

## 🟢 Quick Wins

### 9. **Add MMR (Maximum Marginal Relevance)** Done

- **Problem:** Retrieved chunks may be too similar (redundant)
- **Fix:** Use MMR search type for diversity

```python
retriever = store.as_retriever(
    search_type="mmr",  # Add this
    search_kwargs={
        "k": 6,
        "fetch_k": 20,  # Fetch more, select diverse 6
        "lambda_mult": 0.7  # Balance relevance vs diversity
    }
)
```

- **File:** `vector_store.py`

---

### 10. **Show Retrieved Context in UI (Debug)** Done

- Add option in Streamlit to display the actual retrieved chunks
- Helps diagnose if retrieval or generation is the problem
- **File:** `app.py`

---

## 📋 Implementation Priority

| Priority | Task                    | Impact    | Effort    |
| -------- | ----------------------- | --------- | --------- |
| 1        | Upgrade LLM to 7-8B     | 🔥 High   | ⚡ Low    |
| 2        | Add MMR search          | 🔥 High   | ⚡ Low    |
| 3        | Improve prompts         | 🔥 High   | ⚡ Low    |
| 4        | Increase k to 8-10      | 🔵 Medium | ⚡ Low    |
| 5        | Add reranking           | 🔥 High   | 🔧 Medium |
| 6        | Hybrid search (BM25)    | 🔥 High   | 🔧 Medium |
| 7        | Upgrade embedding       | 🔵 Medium | ⚡ Low    |
| 8        | Query expansion         | 🔵 Medium | 🔧 Medium |
| 9        | Reduce chunk size       | 🔵 Medium | ⚡ Low    |
| 10       | Remove semantic headers | 🟢 Low    | ⚡ Low    |

---

## 🧪 Testing Strategy

After each change, test with these query types:

1. **Factual:** "What is the eligibility for BioCARe programme?"
2. **Comparative:** "Compare Machine Learning basics vs advanced topics"
3. **Specific term:** "What does IYBF stand for?"
4. **Multi-hop:** "Who can apply for fellowships requiring PhD?"
5. **Out of scope:** "What is quantum computing?" (should say don't know)

---

## 🛠️ Suggested Immediate Actions

### Step 1: Quick Config Changes (5 min)

```python
# config.py
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"  # Better reasoning
EMBEDDING_MODEL = "nomic-embed-text:137m-v1.5-fp16"  # Keep for now
```

### Step 2: Add MMR + More Chunks (10 min)

```python
# vector_store.py - query_vector_store()
search_kwargs = {
    "k": 8,
    "fetch_k": 25,
}
retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs=search_kwargs
)
```

### Step 3: Better Prompts (15 min)

Update all prompt templates with structured instructions.

---

## 📦 Optional Dependencies to Add

```txt
# For reranking
sentence-transformers>=2.2.0  # CrossEncoder reranking

# For BM25 hybrid search
rank_bm25>=0.2.2

# Alternative: Cohere rerank (API)
cohere>=4.0
```

---

## Notes

- The current system uses Chroma which is good for small-medium datasets
- For production with >100k docs, consider Qdrant or Weaviate
- Audio transcription (faster-whisper) looks well optimized ✓
- Access control (L1-L5) implementation is solid ✓
- Incremental sync is working correctly ✓

---

_Generated: 2026-01-16_
