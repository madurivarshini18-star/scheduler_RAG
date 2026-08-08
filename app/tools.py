"""
Two tools for the scheduling agent:

- schedule_maker(date, purpose): stores a plan for a given day.
- get_schedule(date): retrieves everything planned for a given day.

Embeddings are generated via Pinecone's hosted inference API
(multilingual-e5-large), so no local torch/GPU is needed.

get_schedule filters by exact `date` metadata rather than relying on
vector similarity — dates are exact tokens, not semantic concepts.
"""

import uuid
from datetime import datetime

from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec

from app.config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def _ensure_index():
    existing = [idx.name for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        # Create index with integrated (hosted) embedding — no local model needed.
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            # Integrated embedding: Pinecone embeds text server-side
            embed={
                "model": PINECONE_EMBEDDING_MODEL,
                "field_map": {"text": "purpose"},
            },
        )
    return _pc.Index(PINECONE_INDEX_NAME)


_index = _ensure_index()


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
    # Upsert with text field — Pinecone embeds it server-side via integrated model
    _index.upsert_records(
        records=[
            {
                "id": vector_id,
                "purpose": purpose,
                "date": norm_date,
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
    # Search with the date string as query text; filter ensures exact-date match
    results = _index.search(
        query={"inputs": {"text": norm_date}, "top_k": 20},
        filter={"date": {"$eq": norm_date}},
        include_metadata=True,
    )
    matches = results.get("matches") or results.get("results") or []
    if not matches:
        return f"No plans found for {norm_date}. That day looks free."

    items = [m.get("metadata", {}).get("purpose", "") for m in matches if m.get("metadata", {}).get("purpose")]
    if not items:
        return f"No plans found for {norm_date}. That day looks free."
    return f"On {norm_date} you have: {'; '.join(items)}."
