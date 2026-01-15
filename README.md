# Scoped RAG

A high-performance multimodal RAG (Retrieval-Augmented Generation) system with Docling OCR support for processing PDFs, Office documents (DOCX, PPTX, XLSX), images, and audio files.

## Features

- **🔍 Docling OCR Engine** - High-performance document processing with table extraction
- **📄 PDF Processing** - OCR-enabled text extraction with table structure recognition
- **📝 Office Documents** - Full support for DOCX, PPTX, XLSX files
- **🖼️ Image Processing** - OCR + Vision captioning using Qwen3-VL via Ollama
- **🎵 Audio Processing** - Speech-to-text using Whisper (HuggingFace)
- **🔎 Vector Search** - FAISS HNSW for high-speed similarity search
- **🌐 Web UI** - Streamlit frontend with chat interface
- **⚡ REST API** - FastAPI backend with Docling support

## Supported File Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS |
| Images | PNG, JPG, JPEG, TIFF, TIF, BMP, WEBP |
| Audio | MP3, WAV, FLAC, M4A, OGG, MPEG, MP4 |

## Tech Stack

| Component | Technology |
|-----------|------------|
| OCR Engine | Docling (with fallback support) |
| Embeddings | Ollama (embeddinggemma) |
| LLM | Ollama (llama3.2) |
| Vision | Ollama (qwen3-vl) |
| Speech-to-Text | Whisper Small EN (HuggingFace) |
| Vector Store | FAISS HNSW |
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

# Install Docling OCR (optional, falls back to basic extraction)
pip install docling docling-core python-docx python-pptx openpyxl
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
# Basic usage
uv run python main.py "your question here"

# Force rebuild index
uv run python main.py --rebuild "your question"

# Use Docling OCR (recommended for Office docs)
uv run python main.py --use-docling --rebuild "your question"
```

### Option 3: Test OCR Module

```bash
uv run python test_ocr.py
```

## Adding Documents

1. Place files in the `data/` folder:
   - PDFs (`.pdf`)
   - Office docs (`.docx`, `.pptx`, `.xlsx`)
   - Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`)
   - Audio (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.mpeg`)

2. Rebuild the index:
   - **Web UI**: Click "🔄 Rebuild Index" in sidebar
   - **CLI**: `uv run python main.py --rebuild --use-docling "test"`
   - **API**: `POST /reindex?use_docling=true`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | File statistics |
| `/supported-formats` | GET | List supported formats |
| `/query` | POST | Query (answer only) |
| `/query-with-sources` | POST | Query with sources |
| `/reindex?use_docling=true` | POST | Rebuild vector index |
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
