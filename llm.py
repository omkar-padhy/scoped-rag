"""
LLM Module with Cascading Fallback for RAG
Primary (70B) → Secondary (17B) → Tertiary (8B) → Local (Ollama)

Uses Groq API with automatic fallback on rate limits.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Check for Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _try_groq_llm(model_name: str):
    """Attempt to create a Groq LLM instance"""
    if not GROQ_API_KEY:
        logger.debug("No GROQ_API_KEY found")
        return None
    
    try:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=2048,
        )
        logger.info(f"✅ Groq LLM ready: {model_name}")
        return llm
    except ImportError:
        logger.warning("langchain-groq not installed. Run: uv add langchain-groq")
        return None
    except Exception as e:
        logger.warning(f"❌ Groq {model_name} init failed: {e}")
        return None


def _get_ollama_llm(model_name: str):
    """Create local Ollama LLM (always works if Ollama is running)"""
    from langchain_ollama import OllamaLLM
    
    logger.info(f"🏠 Using local Ollama: {model_name}")
    return OllamaLLM(model=model_name)


def get_llm():
    """
    Get the best available LLM with cascading fallback:
    1. Primary: llama-3.3-70b-versatile (Groq) - best quality
    2. Secondary: llama-4-scout-17b (Groq) - long duration
    3. Tertiary: llama-3.1-8b-instant (Groq) - reliable backup
    4. Fallback: Local Ollama - always available
    """
    from config import (
        LLM_MODEL_PRIMARY, 
        LLM_MODEL_SECONDARY, 
        LLM_MODEL_TERTIARY,
        LLM_MODEL_LOCAL
    )
    
    # Try Primary (70B - best quality)
    llm = _try_groq_llm(LLM_MODEL_PRIMARY)
    if llm:
        return llm
    
    # Try Secondary (17B - long duration)
    llm = _try_groq_llm(LLM_MODEL_SECONDARY)
    if llm:
        return llm
    
    # Try Tertiary (8B - reliable)
    llm = _try_groq_llm(LLM_MODEL_TERTIARY)
    if llm:
        return llm
    
    # Fallback to local Ollama
    return _get_ollama_llm(LLM_MODEL_LOCAL)


def query_with_fallback(prompt: str) -> str:
    """
    Query LLM with automatic retry on rate limit.
    Cascades: Primary → Secondary → Tertiary → Local
    
    Use this for individual queries where you want automatic
    fallback if one model hits rate limits mid-request.
    """
    from config import (
        LLM_MODEL_PRIMARY, 
        LLM_MODEL_SECONDARY, 
        LLM_MODEL_TERTIARY,
        LLM_MODEL_LOCAL
    )
    
    models_to_try = []
    
    if GROQ_API_KEY:
        models_to_try.append(("groq", LLM_MODEL_PRIMARY))
        models_to_try.append(("groq", LLM_MODEL_SECONDARY))
        models_to_try.append(("groq", LLM_MODEL_TERTIARY))
    
    models_to_try.append(("ollama", LLM_MODEL_LOCAL))
    
    last_error = None
    
    for provider, model_name in models_to_try:
        try:
            if provider == "groq":
                llm = _try_groq_llm(model_name)
                if not llm:
                    continue
            else:
                llm = _get_ollama_llm(model_name)
            
            response = llm.invoke(prompt)
            # Handle both string and AIMessage responses
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                logger.warning(f"⚠️ Rate limited on {model_name}, trying next...")
            else:
                logger.warning(f"❌ Error with {model_name}: {e}")
            last_error = e
            continue
    
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
