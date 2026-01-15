"""
Audio processing using Whisper Small EN (GPU-safe)
Optimized for RTX 3050 4GB VRAM + 8GB RAM
"""

import os
import re
import time
import hashlib
import subprocess
from pathlib import Path

import numpy as np
from typing import Any

# Set ffmpeg path from imageio-ffmpeg BEFORE importing transformers
FFMPEG_PATH = None
try:
    import imageio_ffmpeg

    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ["PATH"]
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
    print(f"✓ Using ffmpeg from: {FFMPEG_PATH}")
except ImportError:
    print("⚠ imageio-ffmpeg not found, using system ffmpeg")
    FFMPEG_PATH = "ffmpeg"

import torch
# Conditionally import WhisperModel or define dummy for type hints type check
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================
# Configuration
# =====================

DATA_PATH = "data"
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mpeg", ".mp4"}

# Faster-Whisper model (GPU-optimized, ~93% of Large accuracy)
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large-v3
COMPUTE_TYPE = "float16"  # Half-precision for VRAM efficiency
BEAM_SIZE = 5  # Improves word accuracy
VAD_FILTER = True  # Removes silence for cleaner output
LANGUAGE = "en"  # Set to None for auto-detect

TARGET_CHUNK_SIZE = 800  # Characters per chunk

# Global model (lazy-loaded)
_whisper_model = None


# =====================
# Metadata Extraction
# =====================

def extract_file_metadata(file_path: Path) -> dict:
    """
    Extract universal metadata from file:
    - Access Level [L1]-[L5] from filename
    - Creation timestamp
    - Keywords from filename
    """
    file_name = file_path.name
    
    # 1. Access Level (Default = 2: Internal)
    access_level = 2
    match = re.search(r"\[L([1-5])\]", file_name, re.IGNORECASE)
    if match:
        access_level = int(match.group(1))

    # 2. Creation timestamp
    try:
        created_at = os.path.getctime(file_path)
    except Exception:
        created_at = time.time()

    # 3. Keywords from filename
    clean_name = re.sub(r"\[L[1-5]\]", "", file_path.stem)
    keywords = [w.lower() for w in re.split(r"[_\-\s]+", clean_name) if len(w) > 2]
    # Remove common junk words
    junk = {"the", "and", "for", "with", "doc", "file", "audio", "recording"}
    keywords = [k for k in keywords if k not in junk]

    return {
        "access_level": access_level,
        "created_at": created_at,
        "keywords": ", ".join(keywords),
    }


# =====================
# Faster-Whisper Loader
# =====================


def get_whisper_model() -> WhisperModel:
    """Lazy-load Faster-Whisper model (GPU-optimized)"""
    global _whisper_model

    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = COMPUTE_TYPE if device == "cuda" else "int8"
        
        print(f"Loading Faster-Whisper ({MODEL_SIZE}) on {device} with {compute}...")
        
        _whisper_model = WhisperModel(
            MODEL_SIZE,
            device=device,
            compute_type=compute,
        )
        
        print(f"✓ Faster-Whisper loaded on {device}")

    return _whisper_model


# =====================
# Transcription
# =====================


def transcribe_audio(audio_path: Path) -> dict:
    """Transcribe one audio file using Faster-Whisper with timestamps"""
    model = get_whisper_model()

    print(f"🎤 Transcribing: {audio_path.name}")
    
    # Extract file metadata (access level, keywords, etc.)
    meta = extract_file_metadata(audio_path)

    try:
        # Transcribe with Faster-Whisper (handles audio loading internally)
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            language=LANGUAGE,
            word_timestamps=False,  # Segment-level is sufficient for RAG
        )

        # Collect segments with timestamps
        raw_segments = []
        text_parts = []
        
        for segment in segments:
            text_parts.append(segment.text)
            raw_segments.append({
                "text": segment.text,
                "timestamp": (segment.start, segment.end),
            })

        transcription = " ".join(text_parts).strip()
        total_duration = info.duration if info else 0.0

    except Exception as e:
        print(f"❌ Error transcribing {audio_path.name}: {e}")
        transcription = "ERROR_TRANSCRIBING_AUDIO"
        raw_segments = []
        total_duration = 0.0

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path),
        "format": audio_path.suffix[1:].upper(),
        "transcription": transcription,
        "raw_segments": raw_segments,
        "total_duration": total_duration,
        "meta": meta,
    }


# =====================
# Audio File Loader
# =====================


def process_audio_files(files: list[Path]) -> list[Document]:
    """Process specific audio files"""
    audio_data = []
    print(f"Processing {len(files)} audio files...")
    
    for audio_path in files:
        if not audio_path.exists():
            print(f"⚠ File not found: {audio_path}")
            continue
            
        result = transcribe_audio(audio_path)
        audio_data.append(result)
        print(f"  ✓ {audio_path.name} ({len(result['transcription'])} chars)")

    if not audio_data:
        return []

    documents = create_audio_documents(audio_data)
    chunks = split_audio_documents(documents)
    return add_audio_chunk_ids(chunks)


def process_audio(data_path: str = DATA_PATH) -> list[Document]:
    """Process all audio files in directory"""
    data_dir = Path(data_path)
    if not data_dir.exists():
        return []
        
    audio_files = [
        f
        for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]
    
    return process_audio_files(audio_files)


def create_audio_documents(audio_data: list[dict]) -> list[Document]:
    """
    Convert transcriptions to LangChain Documents with timestamp-aware chunking.
    Groups Whisper segments into chunks while preserving start/end timestamps.
    """
    documents = []

    for data in audio_data:
        segments = data.get("raw_segments", [])
        meta = data.get("meta", {})
        file_name = data["file_name"]
        total_duration = data.get("total_duration", 0.0)
        access_level = meta.get("access_level", 2)
        keywords = meta.get("keywords", "")
        created_at = meta.get("created_at", 0.0)
        
        # If no segments but we have transcription, fall back to simple approach
        if not segments and data.get("transcription") and data["transcription"] != "ERROR_TRANSCRIBING_AUDIO":
            doc = Document(
                page_content=f"[Audio: {file_name}] [Access: L{access_level}]\n\n{data['transcription']}",
                metadata={
                    "source": file_name,
                    "file_name": file_name,
                    "doc_type": "audio",
                    "format": data.get("format", ""),
                    "access_level": access_level,
                    "created_at": created_at,
                    "keywords": keywords,
                    "duration_total": total_duration,
                    "start_seconds": 0.0,
                    "end_seconds": total_duration,
                }
            )
            documents.append(doc)
            continue
            
        if not segments:
            continue

        # Group segments into chunks (~800 chars) while preserving timestamps
        current_text_buffer = []
        current_start = None
        current_len = 0

        for seg in segments:
            text = seg.get("text", "")
            timestamps = seg.get("timestamp")
            
            if not timestamps or not isinstance(timestamps, (list, tuple)):
                continue
                
            start, end = timestamps[0], timestamps[1] if len(timestamps) > 1 else timestamps[0]
            if start is None: start = 0.0
            if end is None: end = start

            # Initialize block start time
            if current_start is None:
                current_start = start

            current_text_buffer.append(text)
            current_len += len(text)

            # If buffer is big enough, finalize this chunk
            if current_len >= TARGET_CHUNK_SIZE:
                merged_text = "".join(current_text_buffer).strip()
                
                if merged_text:
                    doc = Document(
                        page_content=f"[Time: {current_start:.1f}s - {end:.1f}s] [Access: L{access_level}]\n\n{merged_text}",
                        metadata={
                            "source": file_name,
                            "file_name": file_name,
                            "doc_type": "audio",
                            "format": data.get("format", ""),
                            "access_level": access_level,
                            "created_at": created_at,
                            "keywords": keywords,
                            "duration_total": total_duration,
                            "start_seconds": current_start,
                            "end_seconds": end,
                        }
                    )
                    documents.append(doc)
                
                # Reset buffer
                current_text_buffer = []
                current_start = None
                current_len = 0

        # Process remaining text in buffer
        if current_text_buffer:
            merged_text = "".join(current_text_buffer).strip()
            last_end = segments[-1]["timestamp"][1] if segments and segments[-1].get("timestamp") else total_duration
            if last_end is None: last_end = total_duration
            if current_start is None: current_start = 0.0

            if merged_text:
                doc = Document(
                    page_content=f"[Time: {current_start:.1f}s - {last_end:.1f}s] [Access: L{access_level}]\n\n{merged_text}",
                    metadata={
                        "source": file_name,
                        "file_name": file_name,
                        "doc_type": "audio",
                        "format": data.get("format", ""),
                        "access_level": access_level,
                        "created_at": created_at,
                        "keywords": keywords,
                        "duration_total": total_duration,
                        "start_seconds": current_start,
                        "end_seconds": last_end,
                    }
                )
                documents.append(doc)

    return documents


# =====================
# Chunking for RAG
# =====================


def split_audio_documents(documents: list[Document]) -> list[Document]:
    """
    Pass-through: We now handle chunking inside create_audio_documents
    to preserve timestamp alignment.
    """
    return documents


def add_audio_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Add unique chunk IDs based on content hash"""
    for idx, chunk in enumerate(chunks):
        file_name = chunk.metadata.get("file_name", "unknown")
        start = chunk.metadata.get("start_seconds", 0)
        # Create unique ID from file + position
        chunk.metadata["chunk_id"] = f"{file_name}:{start:.1f}:{idx}"
    return chunks


# =====================
# Full Pipeline
# =====================


# Entry point for processing specific files
def process_audio_files(files: list[Path]) -> list[Document]:
    """Process specific audio files with enhanced metadata"""
    audio_data = []
    print(f"Processing {len(files)} audio files...")
    
    for audio_path in files:
        if not audio_path.exists():
            print(f"⚠ File not found: {audio_path}")
            continue
            
        result = transcribe_audio(audio_path)
        audio_data.append(result)
        
        seg_count = len(result.get("raw_segments", []))
        duration = result.get("total_duration", 0)
        print(f"  ✓ {audio_path.name} ({seg_count} segments, {duration:.1f}s)")

    if not audio_data:
        return []

    documents = create_audio_documents(audio_data)
    return add_audio_chunk_ids(documents)


# =====================
# Entry Point
# =====================

if __name__ == "__main__":
    # Test metadata extraction
    test_path = Path("data/meeting_notes_[L3].mp3")
    print(f"Testing metadata extraction on: {test_path.name}")
    print(extract_file_metadata(test_path))
    
    chunks = process_audio()

    if chunks:
        print("\n--- Sample Output ---")
        for c in chunks[:2]:
            print(f"\nChunk ID: {c.metadata['chunk_id']}")
            print(f"Access Level: {c.metadata.get('access_level')}")
            print(f"Time Range: {c.metadata.get('start_seconds'):.1f}s - {c.metadata.get('end_seconds'):.1f}s")
            print(c.page_content[:200], "...")
