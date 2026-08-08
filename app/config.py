import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


GROQ_API_KEY = _require("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

PINECONE_API_KEY = _require("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "schedule-agent")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Pinecone hosted embedding model — no local torch/GPU required.
# The index must be created with integrated embedding (model name set at index creation).
# multilingual-e5-large outputs 1024 dims; pinecone-sparse-english-v0 for sparse indexes.
PINECONE_EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL", "multilingual-e5-large")
