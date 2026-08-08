# Schedule RAG Agent

An agentic scheduling assistant. You tell it your plans, it stores them in
Pinecone; you ask it things like *"I want to go to the Spiderman movie on
Aug 20 — do I have work that day?"* and it checks your stored schedule
before answering.

**Stack:** LangChain 1.x (`create_agent`) · Groq (LLM) · Pinecone (vector
store) · sentence-transformers (local embeddings) · FastAPI + LangServe
(deployment)

## How it works

Two tools, both defined in `app/tools.py`:

- **`schedule_maker(date, purpose)`** — embeds the purpose text and upserts
  it into Pinecone with `{date, purpose}` metadata.
- **`get_schedule(date)`** — looks up everything stored for that exact date
  via a Pinecone **metadata filter** (not similarity search — dates are
  exact tokens, so filtering is far more reliable than "nearest neighbor").

The agent (`app/agent.py`) resolves relative dates ("tomorrow," "next
Friday") against today's date in its system prompt, then decides which
tool to call.

> **Note on LangServe:** LangChain deprecated LangServe in Nov 2024 in
> favor of LangGraph Platform, though it still receives bug fixes and
> works fine for a project like this. `app/server.py` is a thin FastAPI
> wrapper around a single `RunnableLambda`, so if you ever want to drop
> LangServe, you only need to change a few lines — the agent logic itself
> doesn't depend on it.

---

## 1. Prototype in Colab

1. Get free API keys:
   - Groq: https://console.groq.com/keys
   - Pinecone: https://app.pinecone.io (free tier includes one serverless index)
2. Open `notebooks/colab_prototype.ipynb` in Colab (File → Upload notebook,
   or push this repo to GitHub first and open it directly from there).
3. Run the cells top to bottom. You'll be prompted for your API keys via
   `getpass` (they aren't saved anywhere).
4. Try your own questions in the last few cells — save a few plans, then
   ask conflict-check questions against them.

## 2. Push to GitHub

```bash
cd schedule-rag-agent
git init
git add .
git commit -m "Initial commit: schedule RAG agent"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.env` is already in `.gitignore` — double check it never gets committed,
since it holds your API keys.

## 3. Run locally (optional, before deploying)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # fill in your real keys
uvicorn app.server:app --reload
```

Visit `http://localhost:8000/agent/playground` for an interactive LangServe
UI, or:

```bash
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "I have a team offsite on 2026-08-20"}}'
```

## 4. Deploy to Render

**Option A — Blueprint (recommended):** this repo includes `render.yaml`.
In the Render dashboard: New → Blueprint → connect your GitHub repo →
Render reads `render.yaml` and sets up the service automatically. You'll
just need to fill in `GROQ_API_KEY` and `PINECONE_API_KEY` in the
dashboard (they're marked `sync: false` so Render prompts for them rather
than storing them in the blueprint file).

**Option B — Manual web service:**
1. New → Web Service → connect your GitHub repo.
2. Runtime: Python 3.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `GROQ_API_KEY`, `PINECONE_API_KEY`,
   `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`,
   `EMBEDDING_MODEL` (see `.env.example` for values).

Once deployed, test it:

```bash
curl -X POST https://<your-app>.onrender.com/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "Do I have anything on 2026-08-20?"}}'
```

### A note on Render's free tier

Free web services spin down after ~15 minutes of inactivity and take
30–60 seconds to wake up on the next request — fine for a demo/personal
project, but worth knowing if the first request after idle time seems slow.

---

## Extending this

- **Multi-turn memory:** `create_agent` supports a `checkpointer` (from
  `langgraph.checkpoint`) so conversations can persist across requests via
  a `thread_id`. Currently each request is stateless.
- **Semantic "when am I free" queries:** `get_schedule` only does exact-date
  lookups today. You could add a third tool that does a broader similarity
  search over `purpose` text for fuzzier questions like "when's my next
  dentist appointment."
- **Namespaces per user:** if this becomes multi-user, give each user their
  own Pinecone namespace instead of one shared index.
