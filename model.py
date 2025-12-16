from langchain_ollama import OllamaEmbeddings, OllamaLLM


def get_embedding():
    return OllamaEmbeddings(model="embeddinggemma:300m-bf16")

def get_llm():
    return OllamaLLM(model="llama3.2:3b-instruct-q8_0")

def get_image_description():
    return OllamaLLM(model="qwen3-vl:2b-instruct-q4_K_M")
