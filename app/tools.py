"""
Two tools for the scheduling agent:

- schedule_maker(date, purpose): stores a plan for a given day.
- get_schedule(date): retrieves everything planned for a given day.

Design notes:
1. get_schedule does NOT rely on vector similarity to find the right day.
   Dates are exact tokens, not semantic concepts, so similarity search over
   them is unreliable ("2026-08-20" isn't meaningfully "close to"
   "2026-08-21"). Instead we filter Pinecone by an exact `date` metadata
   field and only use the embedding to satisfy Pinecone's query API.
2. We talk to Pinecone via its native SDK (not langchain-pinecone), since
   that wrapper doesn't yet support langchain-core 1.x. This also means
   embeddings are computed directly with sentence-transformers.
"""

import uuid
from datetime import datetime

from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)
_model = SentenceTransformer(EMBEDDING_MODEL)


def _ensure_index():
    existing = [idx.name for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return _pc.Index(PINECONE_INDEX_NAME)


_index = _ensure_index()


def _embed(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()


def _normalize_date(date_str: str) -> str:
    """Accepts a handful of common formats and normalizes to YYYY-MM-DD.
    The agent is instructed (see agent.py system prompt) to always pass
    ISO dates, but this keeps the tool safe if it doesn't."""
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str  # fall back to whatever the model gave us


@tool
def schedule_maker(date: str, purpose: str) -> str:
    """Save a plan/commitment for a specific date.

    Args:
        date: The date in YYYY-MM-DD format (e.g. "2026-08-20").
        purpose: What the person is doing that day (e.g. "team offsite",
            "dentist appointment 3pm").

    Returns:
        A confirmation message.
    """
    norm_date = _normalize_date(date)
    vector_id = f"{norm_date}-{uuid.uuid4().hex[:8]}"
    _index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": _embed(purpose),
                "metadata": {"date": norm_date, "purpose": purpose},
            }
        ]
    )
    return f"Saved: on {norm_date} you have '{purpose}'."


@tool
def get_schedule(date: str) -> str:
    """Look up everything planned for a specific date.

    Args:
        date: The date in YYYY-MM-DD format (e.g. "2026-08-20").

    Returns:
        A description of what's planned that day, or a message saying
        the day is free if nothing is found.
    """
    norm_date = _normalize_date(date)
    result = _index.query(
        vector=_embed(norm_date),
        top_k=20,
        filter={"date": {"$eq": norm_date}},
        include_metadata=True,
    )
    matches = result.matches
    if not matches:
        return f"No plans found for {norm_date}. That day looks free."

    items = [m["metadata"]["purpose"] for m in matches]
    joined = "; ".join(items)
    return f"On {norm_date} you have: {joined}."
