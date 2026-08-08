"""
Two tools for the scheduling agent:

- schedule_maker(date, purpose): stores a plan for a given day.
- get_schedule(date): retrieves everything planned for a given day.

Embeddings use fastembed (ONNX Runtime, no torch/CUDA) — lightweight enough
for Render's free tier 512MB RAM limit.

get_schedule filters by exact `date` metadata rather than relying on
vector similarity — dates are exact tokens, not semantic concepts.
"""

import uuid
from datetime import datetime

from fastembed import TextEmbedding
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)
_embedder = TextEmbedding(model_name=EMBEDDING_MODEL)


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
    # fastembed returns a generator of numpy arrays
    return list(_embedder.embed([text]))[0].tolist()


def _normalize_date(date_str: str) -> str:
    """Normalize common date formats to YYYY-MM-DD."""
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


@tool
def schedule_maker(date: str, purpose: str) -> str:
    """Save a plan/commitment for a specific date.

    Args:
        date: The date in YYYY-MM-DD format (e.g. "2026-08-20").
        purpose: What the person is doing that day (e.g. "team offsite").

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

    items = [m.metadata["purpose"] for m in matches if m.metadata.get("purpose")]
    if not items:
        return f"No plans found for {norm_date}. That day looks free."
    return f"On {norm_date} you have: {'; '.join(items)}."
