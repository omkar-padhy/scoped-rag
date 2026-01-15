from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config import EMBEDDING_MODEL, VISION_MODEL

# Import cascading LLM from new module
from llm import get_llm, query_with_fallback


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


# get_llm is now imported from llm.py with cascading fallback


def get_image_description():
    return OllamaLLM(model=VISION_MODEL)
