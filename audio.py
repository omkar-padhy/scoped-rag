"""
Audio processing using Whisper Small EN (GPU-safe)
Optimized for RTX 3050 4GB VRAM + 8GB RAM
"""

import os
import subprocess
import numpy as np
from pathlib import Path

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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =====================
# Configuration
# =====================

DATA_PATH = "data"
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mpeg", ".mp4"}

MODEL_ID = "openai/whisper-small.en"
CHUNK_LENGTH_S = 30   # prevents VRAM spikes
BATCH_SIZE = 1        # safe for 4GB VRAM

# Global pipeline (lazy-loaded)
_whisper_pipe = None


# =====================
# Whisper Loader
# =====================

def get_whisper_pipeline():
    """Lazy-load Whisper Small EN pipeline"""
    global _whisper_pipe

    if _whisper_pipe is None:
        print("Loading Whisper Small EN model...")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        _whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=torch_dtype,
        )

        print(f"Whisper loaded on {device}")

    return _whisper_pipe


# =====================
# Audio Loading (using ffmpeg directly)
# =====================

def load_audio_with_ffmpeg(audio_path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file using ffmpeg and return numpy array"""
    cmd = [
        FFMPEG_PATH,
        "-i", str(audio_path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
    
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio


# =====================
# Transcription
# =====================

def transcribe_audio(audio_path: Path) -> dict:
    """Transcribe one audio file"""
    pipe = get_whisper_pipeline()

    print(f"Transcribing: {audio_path.name}")

    try:
        # Load audio using our ffmpeg wrapper
        audio_array = load_audio_with_ffmpeg(audio_path)
        
        # Pass numpy array instead of file path
        # Note: whisper-small.en is English-only, no language/task needed
        result = pipe(
            {"raw": audio_array, "sampling_rate": 16000},
            return_timestamps=True,
        )

        transcription = result["text"].strip()
        chunks = result.get("chunks", [])

        duration_info = ""
        if chunks:
            last = chunks[-1].get("timestamp", [None, None])[1]
            if last:
                duration_info = f"~{last:.1f}s"

    except Exception as e:
        print(f"❌ Error transcribing {audio_path.name}: {e}")
        transcription = "ERROR_TRANSCRIBING_AUDIO"
        duration_info = ""

    return {
        "file_name": audio_path.name,
        "format": audio_path.suffix[1:].upper(),
        "transcription": transcription,
        "duration_info": duration_info,
    }


# =====================
# Audio File Loader
# =====================

def load_audio_files(data_path: str = DATA_PATH) -> list[dict]:
    """Load and transcribe all supported audio files"""
    data_dir = Path(data_path)
    audio_data = []

    if not data_dir.exists():
        print(f"❌ Data path not found: {data_path}")
        return audio_data

    audio_files = [
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    print(f"Found {len(audio_files)} audio files")

    for audio_path in audio_files:
        result = transcribe_audio(audio_path)
        audio_data.append(result)
        print(f"  ✓ {audio_path.name} ({len(result['transcription'])} chars)")

    return audio_data


# =====================
# LangChain Documents
# =====================

def create_audio_documents(audio_data: list[dict]) -> list[Document]:
    """Convert transcriptions to LangChain Documents"""
    documents = []

    for data in audio_data:
        if data["transcription"] == "ERROR_TRANSCRIBING_AUDIO":
            continue

        content = "\n".join([
            f"[Audio File: {data['file_name']}]",
            f"[Format: {data['format']}]",
            f"[Duration: {data['duration_info']}]",
            "",
            "=== Transcription ===",
            data["transcription"]
        ])

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": data["file_name"],
                    "format": data["format"],
                    "type": "audio",
                    "duration": data["duration_info"]
                }
            )
        )

    return documents


# =====================
# Chunking for RAG
# =====================

def split_audio_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def add_audio_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_source = None
    idx = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")

        if source == last_source:
            idx += 1
        else:
            idx = 0

        chunk.metadata["chunk_id"] = f"{source}:{idx}"
        last_source = source

    return chunks


# =====================
# Full Pipeline
# =====================

def process_audio(data_path: str = DATA_PATH) -> list[Document]:
    audio_data = load_audio_files(data_path)
    if not audio_data:
        return []

    documents = create_audio_documents(audio_data)
    chunks = split_audio_documents(documents)
    return add_audio_chunk_ids(chunks)


# =====================
# Entry Point
# =====================

if __name__ == "__main__":
    chunks = process_audio()

    if chunks:
        print("\n--- Sample Output ---")
        for c in chunks[:2]:
            print(f"\nChunk ID: {c.metadata['chunk_id']}")
            print(c.page_content[:200], "...")
