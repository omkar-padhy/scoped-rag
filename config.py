import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
DATA_PATH = Path("data")
DB_PATH = Path("chroma_db")

# ===================
# LLM Models (Groq API with cascading fallback)
# ===================
# 1. Best Quality: 70B params - smartest reasoning
LLM_MODEL_PRIMARY = "llama-3.3-70b-versatile"
# 2. Long Duration: 30K tokens/min, 500K tokens/day
LLM_MODEL_SECONDARY = "meta-llama/llama-4-scout-17b-16e-instruct"
# 3. OK Fallback: 14.4K requests/day - most reliable
LLM_MODEL_TERTIARY = "llama-3.1-8b-instant"
# 4. Local: Always available offline
LLM_MODEL_LOCAL = "llama3.2:3b-instruct-q4_K_M"

# Default (used if no API)
LLM_MODEL = LLM_MODEL_LOCAL

# ===================
# Embedding Model
# ===================
EMBEDDING_MODEL = "mxbai-embed-large:335m-v1-fp16"

# Vision Models
# Primary: OpenRouter (requires OPENROUTER_API_KEY env var)
VISION_MODEL_PRIMARY = "nvidia/nemotron-nano-12b-v2-vl:free"
VISION_MODEL_SECONDARY = "qwen/qwen-2.5-vl-7b-instruct:free"
# Fallback: Local Ollama
VISION_MODEL_LOCAL = "qwen3-vl:2b-instruct-q4_K_M"

# Default reference
VISION_MODEL = VISION_MODEL_LOCAL

# Audio transcription (Faster-Whisper)
# Options: tiny, base, small, medium, large-v3
WHISPER_MODEL = "small.en"  # Best English accuracy for 4GB VRAM
