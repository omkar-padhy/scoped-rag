# Scoped RAG

A multimodal RAG (Retrieval-Augmented Generation) system that processes PDFs, images, and audio files.

## Features

- **PDF Processing** - Extract and chunk text from PDF documents
- **Image Processing** - OCR + description using Qwen3-VL via Ollama
- **Audio Processing** - Speech-to-text using Whisper (HuggingFace)
- **Vector Search** - FAISS for similarity search
- **Web UI** - Streamlit frontend with chat interface
- **REST API** - FastAPI backend

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Ollama (embeddinggemma) |
| LLM | Ollama (llama3.2) |
| Vision | Ollama (qwen3-vl) |
| Speech-to-Text | Whisper Small EN (HuggingFace) |
| Vector Store | FAISS |
| Backend | FastAPI |
| Frontend | Streamlit |

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- [Ollama](https://ollama.ai/) running locally

### Required Ollama Models

```bash
ollama pull embeddinggemma:300m-bf16
ollama pull llama3.2:3b-instruct-q8_0
ollama pull qwen3-vl:2b-instruct-q4_K_M
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Omkar-888/scoped-rag.git
cd scoped-rag

# Install dependencies
uv sync
```

## Usage

### Option 1: Web UI (Recommended)

**Terminal 1 - Start Backend:**
```bash
uv run python server.py
```

**Terminal 2 - Start Frontend:**
```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Option 2: CLI

```bash
uv run python main.py "your question here"
```

## Adding Documents

1. Place files in the `data/` folder:
   - PDFs (`.pdf`)
   - Images (`.png`, `.jpg`, `.jpeg`)
   - Audio (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.mpeg`)

2. Rebuild the index:
   - **Web UI**: Click "🔄 Rebuild Index" in sidebar
   - **CLI**: Delete `faiss_index/` folder and run again

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Query (answer only) |
| `/query-with-sources` | POST | Query with sources |
| `/reindex` | POST | Rebuild vector index |
| `/upload` | POST | Upload file to data folder |
| `/files` | GET | List files in data folder |
| `/files/{filename}` | DELETE | Delete a file |

## Project Structure

```
scoped-rag/
├── app.py           # Streamlit frontend
├── server.py        # FastAPI backend
├── main.py          # CLI entry point
├── model.py         # Ollama model config
├── text.py          # PDF processing
├── image.py         # Image OCR + description
├── audio.py         # Audio transcription
├── vector_store.py  # FAISS operations
├── data/            # Source documents
├── faiss_index/     # Vector index storage
└── pyproject.toml   # Dependencies
```

## License

MIT
