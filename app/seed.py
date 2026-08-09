"""
Seed sample schedule data for the next 30 days.
Run once: python -m app.seed
Or called automatically on first startup via a seeded flag in Pinecone metadata.
"""

import uuid
from datetime import datetime, timedelta

from fastembed import TextEmbedding
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

SAMPLE_EVENTS = [
    # (day_offset, time, title, event_type, description)
    (0,  "09:00", "Daily Standup",              "meeting",     "Team sync — blockers and progress"),
    (0,  "14:00", "Code Review Session",         "task",        "Review PRs from the sprint"),
    (1,  "10:00", "Product Roadmap Meeting",      "meeting",     "Q3 roadmap planning with PM"),
    (1,  "15:30", "Doctor Appointment",           "appointment", "Annual health check-up"),
    (2,  "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (2,  "13:00", "Lunch with Client",            "appointment", "Business lunch at The Grand"),
    (3,  "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (3,  "11:00", "Python Workshop",             "workshop",    "Advanced Python patterns — 2 hours"),
    (3,  "16:00", "Budget Review",               "meeting",     "Monthly budget review with finance"),
    (4,  "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (4,  "14:00", "UI/UX Design Review",          "meeting",     "Review new design mockups"),
    (5,  "10:00", "Sprint Planning",             "meeting",     "Plan tasks for next sprint"),
    (7,  "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (7,  "11:00", "Security Training",           "workshop",    "Mandatory cybersecurity awareness"),
    (8,  "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (8,  "14:00", "1-on-1 with Manager",         "meeting",     "Weekly check-in and feedback"),
    (9,  "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (9,  "10:30", "System Architecture Review",   "meeting",     "Review proposed microservices arch"),
    (10, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (10, "13:00", "Dentist Appointment",          "appointment", "Teeth cleaning"),
    (10, "15:00", "Performance Review Prep",      "task",        "Prepare self-assessment document"),
    (11, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (11, "14:00", "Deployment Window",           "task",        "Deploy v2.1 to production"),
    (12, "10:00", "Sprint Retrospective",        "meeting",     "End of sprint retro"),
    (14, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (14, "11:00", "Machine Learning Workshop",   "workshop",    "Intro to LangChain agents — 3 hours"),
    (15, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (15, "15:00", "Stakeholder Demo",            "meeting",     "Demo new features to stakeholders"),
    (16, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (16, "13:00", "Team Lunch",                  "appointment", "Team bonding lunch"),
    (17, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (17, "14:30", "Quarterly Business Review",   "meeting",     "QBR with leadership team"),
    (18, "10:00", "Agile Coaching Session",      "workshop",    "Scrum master workshop"),
    (21, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (21, "11:00", "API Integration Review",      "meeting",     "Review third-party API contracts"),
    (22, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (22, "14:00", "Tax Filing Deadline",         "task",        "Submit quarterly tax documents"),
    (23, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (23, "10:00", "Cloud Cost Optimization",     "meeting",     "Review and reduce AWS spend"),
    (24, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (24, "15:00", "Gym Session",                 "appointment", "Personal fitness session"),
    (25, "09:00", "Weekly Wrap-up",              "meeting",     "End of week team sync"),
    (25, "11:00", "Data Privacy Workshop",       "workshop",    "GDPR compliance training"),
    (28, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (28, "13:00", "Product Launch Prep",         "meeting",     "Final prep for product launch"),
    (29, "09:00", "Daily Standup",               "meeting",     "Team sync"),
    (29, "14:00", "Product Launch Day",          "meeting",     "Go-live — all hands on deck"),
]


def seed():
    existing = [idx.name for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    index = _pc.Index(PINECONE_INDEX_NAME)

    # Check if already seeded
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0) if isinstance(stats, dict) else getattr(stats, "total_vector_count", 0)
    if total > 10:
        print(f"Index already has {total} vectors — skipping seed.")
        return

    today = datetime.now().date()
    vectors = []

    for offset, time, title, event_type, description in SAMPLE_EVENTS:
        event_date = (today + timedelta(days=offset)).strftime("%Y-%m-%d")
        text = f"{event_type} on {event_date} at {time}: {title}. {description}"
        embedding = list(_embedder.embed([text]))[0].tolist()
        vectors.append({
            "id": f"seed-{event_date}-{uuid.uuid4().hex[:6]}",
            "values": embedding,
            "metadata": {
                "date": event_date,
                "time": time,
                "title": title,
                "event_type": event_type,
                "description": description,
            },
        })

    # Upsert in batches of 50
    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i:i+50])

    print(f"Seeded {len(vectors)} sample events into '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    seed()
