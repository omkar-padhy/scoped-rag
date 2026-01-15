"""FastAPI backend for RAG system with Docling OCR support"""

import os
import json
import hashlib
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query as QueryParam, UploadFile
from pydantic import BaseModel

from audio import process_audio_files
from config import DATA_PATH, DB_PATH
from ocr import (
    process_documents,
    process_specific_files,
    process_images_with_ocr_and_caption,
    get_file_stats,
    get_supported_extensions,
)
from vector_store import (
    create_vector_store,
    load_vector_store,
    query_vector_store,
    query_with_sources,
    save_vector_store,
    add_documents_to_store,
    delete_documents_by_filename
)

MANIFEST_PATH = Path(DB_PATH) / "manifest.json"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mpeg", ".mp4"}

app = FastAPI(
    title="Scoped RAG API",
    description="RAG API with Docling OCR support for PDF, DOCX, PPTX, XLSX, and Images",
    version="2.0.0",
)

# Global store and settings
store = None
USE_DOCLING = os.environ.get("USE_DOCLING", "true").lower() == "true"


class Query(BaseModel):
    question: str
    user_level: int = 5


class QueryResponse(BaseModel):
    answer: str


class QueryWithSourcesResponse(BaseModel):
    answer: str
    sources: list[str]
    context: str


class ReindexRequest(BaseModel):
    use_docling: bool = True


class FileStats(BaseModel):
    total: int
    by_type: dict


# ---------------------------------------------------------------------------
# Index Synchronization (Incremental)
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text())
        if "files" not in data:
            return {}  # Assume old format, force rebuild/resync
        return data
    except Exception:
        return {}


def _save_manifest_json(data: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))


def get_current_file_states() -> dict:
    """Get current state of all files in DATA_PATH"""
    if not DATA_PATH.exists():
        return {}
    
    states = {}
    exts = set(get_supported_extensions()) | AUDIO_EXTENSIONS
    
    for f in DATA_PATH.iterdir():
        if f.is_file() and f.suffix.lower() in exts:
            stat = f.stat()
            states[f.name] = {
                "mtime": stat.st_mtime_ns,
                "size": stat.st_size
            }
    return states


def sync_index():
    """Synchronize vector store with file system incrementally"""
    global store
    
    print("Checking for file updates...")
    
    manifest = _load_manifest()
    last_files = manifest.get("files", {})
    current_files = get_current_file_states()
    
    current_filenames = set(current_files.keys())
    last_filenames = set(last_files.keys())
    
    # Identify changes
    to_add = list(current_filenames - last_filenames)
    to_remove = list(last_filenames - current_filenames)
    to_update = []
    
    for fname in current_filenames & last_filenames:
        if current_files[fname] != last_files[fname]:
            to_update.append(fname)
            
    if not to_add and not to_update and not to_remove:
        print("No changes detected.")
        if store is None:
             try:
                 store = load_vector_store()
             except Exception:
                 store = create_vector_store([])
        return store

    print(f"Syncing Index: +{len(to_add)} added, ~{len(to_update)} updated, -{len(to_remove)} removed")
    
    # Init store if needed
    if store is None:
        try:
            store = load_vector_store()
        except Exception:
            store = create_vector_store([])
            
    # Apply removes
    for fname in to_remove:
        delete_documents_by_filename(store, fname)
        
    # Apply updates (Remove then Add)
    for fname in to_update:
        delete_documents_by_filename(store, fname)
        to_add.append(fname)
        
    # Apply adds
    if to_add:
        doc_files = []
        audio_files = []
        
        for fname in to_add:
            fpath = DATA_PATH / fname
            if fpath.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(fpath)
            else:
                doc_files.append(fpath)
        
        new_docs = []
        if doc_files:
            docs = process_specific_files(doc_files)
            new_docs.extend(docs)
            
        if audio_files:
            docs = process_audio_files(audio_files)
            new_docs.extend(docs)
            
        if new_docs:
            add_documents_to_store(store, new_docs)
            
    # Save manifest
    manifest["files"] = current_files
    _save_manifest_json(manifest)
    
    return store


def get_store():
    """Load or create vector store; sync if data changed."""
    return sync_index()


import asyncio

@app.on_event("startup")
async def startup():
    """Load vector store on startup and start auto-sync"""
    get_store()
    asyncio.create_task(watch_for_changes())

async def watch_for_changes():
    """Background task to sync index every 5 seconds"""
    print("Started background file watcher...")
    while True:
        try:
            await asyncio.sleep(5)
            # Run sync in thread to avoid blocking main loop
            await asyncio.to_thread(sync_index)
        except Exception as e:
            print(f"Error in background sync: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "use_docling": USE_DOCLING}


@app.get("/stats", response_model=FileStats)
def get_stats():
    """Get file statistics"""
    stats = get_file_stats(str(DATA_PATH))
    total = stats.pop("total", 0)
    return FileStats(total=total, by_type=stats)


@app.get("/supported-formats")
def supported_formats():
    """Get supported file formats"""
    return {"extensions": get_supported_extensions()}


@app.post("/query", response_model=QueryResponse)
def query(q: Query):
    """Query the RAG system with access level control"""
    try:
        s = get_store()
        answer = query_vector_store(s, q.question, user_level=q.user_level)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-with-sources", response_model=QueryWithSourcesResponse)
def query_sources(q: Query):
    """Query with source documents and access level control"""
    try:
        s = get_store()
        result = query_with_sources(s, q.question, user_level=q.user_level)
        return QueryWithSourcesResponse(
            answer=result["answer"],
            sources=result["sources"],
            context=result["context"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
def reindex(use_docling: bool = QueryParam(default=True)):
    """Synchronize vector index with data files (Incremental)"""
    global store
    try:
        # We ignore use_docling to prefer incremental sync which uses docling/standard processors
        store = sync_index()
        return {
            "status": "synchronized",
            "method": "incremental",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the data folder"""
    try:
        # Get all supported extensions from OCR module
        allowed_ext = set(get_supported_extensions()) | AUDIO_EXTENSIONS
        
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_ext:
            raise HTTPException(400, f"File type {ext} not allowed. Use: {sorted(allowed_ext)}")

        # Save file
        DATA_PATH.mkdir(exist_ok=True)
        file_path = DATA_PATH / file.filename

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"status": "uploaded", "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
def list_files():
    """List files in data folder"""
    try:
        if not DATA_PATH.exists():
            return {"files": []}
        files = [f.name for f in DATA_PATH.iterdir() if f.is_file()]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Delete a file from data folder"""
    try:
        file_path = DATA_PATH / filename
        if not file_path.exists():
            raise HTTPException(404, "File not found")
        file_path.unlink()
        return {"status": "deleted", "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
