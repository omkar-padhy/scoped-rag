"""FastAPI backend for RAG system"""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from audio import process_audio
from text import process_pdfs

DATA_PATH = Path("data")
from image import process_images
from vector_store import (
    create_vector_store,
    load_vector_store,
    query_vector_store,
    query_with_sources,
    save_vector_store,
)

app = FastAPI(title="Scoped RAG API")

# Global store
store = None


class Query(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class QueryWithSourcesResponse(BaseModel):
    answer: str
    sources: list[str]
    context: str


def get_store():
    """Load or create vector store"""
    global store
    if store is None:
        try:
            store = load_vector_store()
        except:
            print("Creating new index...")
            pdf_chunks = process_pdfs()
            image_chunks = process_images()
            audio_chunks = process_audio()
            all_chunks = pdf_chunks + image_chunks + audio_chunks
            store = create_vector_store(all_chunks)
            save_vector_store(store)
    return store


@app.on_event("startup")
async def startup():
    """Load vector store on startup"""
    get_store()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(q: Query):
    """Query the RAG system"""
    try:
        s = get_store()
        answer = query_vector_store(s, q.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-with-sources", response_model=QueryWithSourcesResponse)
def query_sources(q: Query):
    """Query with source documents"""
    try:
        s = get_store()
        result = query_with_sources(s, q.question)
        return QueryWithSourcesResponse(
            answer=result["answer"],
            sources=result["sources"],
            context=result["context"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
def reindex():
    """Rebuild the vector index"""
    global store
    try:
        pdf_chunks = process_pdfs()
        image_chunks = process_images()
        audio_chunks = process_audio()
        all_chunks = pdf_chunks + image_chunks + audio_chunks
        store = create_vector_store(all_chunks)
        save_vector_store(store)
        return {"status": "reindexed", "chunks": len(all_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the data folder"""
    try:
        # Validate file type
        allowed_ext = {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".mp3",
            ".wav",
            ".flac",
            ".m4a",
            ".ogg",
            ".mpeg",
            ".mp4",
        }
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_ext:
            raise HTTPException(400, f"File type {ext} not allowed. Use: {allowed_ext}")

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
