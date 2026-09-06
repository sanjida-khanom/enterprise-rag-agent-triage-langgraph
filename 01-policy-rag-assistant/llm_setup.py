"""
Swappable LLM + embedding provider.

Why this file exists (and why it's a good thing to point at in an interview):
an enterprise shouldn't be hard-wired to one model vendor. Routing every call
through one factory means switching providers is a config change, not a rewrite.
It is also the practical answer to "the venue wifi died" -- set
LLM_PROVIDER=ollama and everything runs locally.

Providers:
  gemini  - Google AI Studio free tier. Default. Needs GOOGLE_API_KEY.
  groq    - Very fast, free tier, open models. Needs GROQ_API_KEY.
  ollama  - Fully local, no internet. Needs `ollama serve` running.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Multilingual by default: Robi's customers write in Bangla, English and
# "Banglish". An English-only embedding model (all-MiniLM-L6-v2) silently
# fails on Bangla queries -- it returns results, they're just wrong.
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def get_llm(temperature: float = 0.0):
    """Return a chat model. temperature=0 by default: for RAG and extraction
    we want reproducible answers, not creative ones."""
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
        )

    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )

    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{PROVIDER}'. Use gemini, groq or ollama."
    )


def get_embeddings():
    """Local sentence-transformers embeddings.

    Deliberately local rather than an API: embeddings run over every chunk of
    every document, so an API-based embedder is the line item that makes a RAG
    pilot expensive. Running them locally is free and keeps document text off
    third-party servers -- which matters when the documents are internal policy.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

def to_text(response) -> str:
    """Normalise any model response into a plain string.

    Newer Gemini models return content as a list of typed blocks (text,
    thinking signatures, and so on) rather than a plain string, and older
    models return a string. Routing every response through one helper means
    a provider changing its response shape is a one-line fix here rather
    than a bug in every caller.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content)
def describe() -> str:
    return f"LLM provider: {PROVIDER} | Embeddings: {EMBED_MODEL}"


if __name__ == "__main__":
    print(describe())
    llm = get_llm()
    print("LLM OK ->", to_text(llm.invoke("Reply with the single word: ready")))
    emb = get_embeddings()
    v = emb.embed_query("balance check")
    print(f"Embeddings OK -> {len(v)} dimensions")
