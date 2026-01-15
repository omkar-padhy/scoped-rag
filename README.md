# Scoped RAG

A high-performance, production-ready Multimodal RAG (Retrieval-Augmented Generation) system featuring **Hybrid Search**, **Reranking**, and an **LLM Cascade** architecture. Supports Docling OCR for complex documents, Vision analysis for images, and GPU-accelerated audio transcription.

## Key Capabilities

- **🧠 Smart Retrieval** - Hybrid search (BM25 + Vector), Query Expansion, and Cross-Encoder Reranking
- **🔄 LLM Cascade** - Automatic fallback: Groq 70B (Quality) → 17B (Speed) → 8B (Reliability) → Local Ollama (Offline)
- **🔍 Docling OCR** - High-fidelity PDF & Office document processing with table structure preservation
- **🖼️ Vision Intelligence** - Dual-stage analysis using OpenRouter (Nemotron/Qwen) with local fallback
- **🎵 Audio Transcription** - GPU-accelerated Speech-to-Text using Faster-Whisper
- **🔒 Access Control** - Granular access levels (L1-L5) for document security
- **⚡ Modern Stack** - ChromaDB, LangChain, Streamlit, and FastAPI

## Supported File Formats

| Category | Formats | Processing Method |
|----------|---------|-------------------|
| Documents | PDF, DOCX, PPTX, XLSX | Docling OCR + Semantic Chunking |
| Text | TXT, MD, CSV, JSON | Recursive Splitting |
| Images | PNG, JPG, WEBP, BMP | Vision Model Captioning + OCR |
| Audio | MP3, WAV, M4A, OGG | Faster-Whisper Transcription |

## Tech Stack

| Component | Technology | Model / Config |
|-----------|------------|----------------|
| **LLM Engine** | Groq API + Ollama | Llama 3.3 70B → 17B → 8B → Llama 3.2 (Local) |
| **Embeddings** | Ollama | `mxbai-embed-large:335m-v1-fp16` |
| **Vector Store** | ChromaDB | Local persistent storage with MMR search |
| **Search** | Hybrid | Vector (Semantic) + BM25 (Keyword) + Reranking |
| **Vision** | OpenRouter/Ollama | Nemotron-12B / Qwen-2.5-VL / Qwen2-VL (Local) |
| **Audio** | Faster-Whisper | `small.en` (GPU optimized) |
| **Frontend** | Streamlit | Chat UI with Source & Context Viewer |

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.ai/) running locally for embeddings/fallback

### Environment Variables (.env)
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
GOOGLE_API_KEY=... (Optional for Gemini)
```

### Required Ollama Models
```bash
ollama pull mxbai-embed-large:335m-v1-fp16
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull qwen3-vl:2b-instruct-q4_K_M
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Omkar-888/scoped-rag.git
cd scoped-rag

# Install dependencies with uv
uv sync

# OR with pip
pip install -r requirements.txt
```

## Usage

### 1. Start the Web UI (Recommended)
This launches both the backend services and the Streamlit frontend.

```bash
uv run streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 2. Interface Features
- **Creating Index:** Upload files via the "📁 Data Management" sidebar tab.
- **RAG Chat:** Ask questions in the main chat window.
- **Debug View:** Check "🔍 Show retrieved context" to see exactly what chunks were sent to the LLM.
- **Rebuild Index:** Use "🔄 Rebuild Index" if you add files manually to `data/`.

### 3. CLI Usage

```bash
# Ask a question via CLI
uv run python main.py "What is the summary of the financial report?"

# Force rebuild the index
uv run python main.py --rebuild "your question"

# Run Quality/Smoke Tests
uv run python test.py
```

## Retrieval Pipeline

The system uses a sophisticated 4-step retrieval process:
1. **Query Expansion:** Generates 3 source queries from your question.
2. **Hybrid Search:**
   - **Vector:** Semantic search using embeddings (MMR for diversity).
   - **BM25:** Keyword-based search for exact matches.
3. **Deduplication:** Merges results from all queries and search methods.
4. **Reranking:** Cross-encoder scoring to select the best 6 chunks.

## Project Structure

```
scoped-rag/
├── app.py           # Streamlit frontend & UI logic
├── config.py        # Central configuration (Models, API Keys)
├── llm.py           # Cascading LLM handler (Groq -> Local)
├── vector_store.py  # ChromaDB, Hybrid Search, Reranking logic
├── ocr.py           # Docling OCR & Chunking
├── server.py        # FastAPI Access Point
├── image.py         # Vision processing
├── audio.py         # Whisper transcription
├── data/            # Input documents directory
└── chroma_db/       # Persistent vector database
```

## License

MIT
