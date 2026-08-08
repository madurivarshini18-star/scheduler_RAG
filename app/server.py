import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.agent import run_agent

app = FastAPI(
    title="Schedule RAG Agent",
    version="1.0",
    description="Agentic scheduling assistant backed by Pinecone + Groq",
)

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentInput(BaseModel):
    input: str


class AgentOutput(BaseModel):
    output: str


@app.post("/agent/invoke", response_model=AgentOutput)
def agent_invoke(payload: AgentInput) -> AgentOutput:
    """Synchronous invoke — returns the final answer as JSON."""
    return AgentOutput(output=run_agent(payload.input))


@app.post("/agent/stream")
def agent_stream(payload: AgentInput) -> StreamingResponse:
    """Server-Sent Events stream — emits one JSON event with the final answer."""

    def _generate():
        output = run_agent(payload.input)
        data = json.dumps({"output": output})
        yield f"data: {data}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agent/playground", response_class=HTMLResponse)
def playground():
    """Schedule entry UI — collects date + activity, silently saves, resets for next entry."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Schedule Planner</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #f0f4ff;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0;
    }
    .card {
      background: #fff;
      border-radius: 16px;
      padding: 40px 36px;
      width: 100%;
      max-width: 440px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.10);
    }
    h1 { font-size: 1.5rem; margin: 0 0 4px; color: #1a1a2e; }
    .subtitle { color: #888; font-size: 0.9rem; margin: 0 0 28px; }
    label { display: block; font-size: 0.85rem; font-weight: 600; color: #444; margin-bottom: 6px; }
    input[type="date"], input[type="text"] {
      width: 100%;
      padding: 11px 14px;
      border: 1.5px solid #dde1f0;
      border-radius: 8px;
      font-size: 1rem;
      color: #1a1a2e;
      outline: none;
      transition: border-color 0.2s;
      margin-bottom: 20px;
    }
    input:focus { border-color: #4f6ef7; }
    button {
      width: 100%;
      padding: 12px;
      background: #4f6ef7;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover:not(:disabled) { background: #3a57e8; }
    button:disabled { background: #b0bcf7; cursor: not-allowed; }
    .status {
      margin-top: 16px;
      text-align: center;
      font-size: 0.9rem;
      min-height: 22px;
      color: #4f6ef7;
    }
    .status.error { color: #e53e3e; }
  </style>
</head>
<body>
  <div class="card">
    <h1>📅 Schedule Planner</h1>
    <p class="subtitle">Add your plans for the day</p>
    <form id="form">
      <label for="date">Date</label>
      <input type="date" id="date" required />
      <label for="activity">What are you doing?</label>
      <input type="text" id="activity" placeholder="e.g. Team offsite, Dentist at 3pm" autocomplete="off" required />
      <button id="btn" type="submit">Save Plan</button>
    </form>
    <div class="status" id="status"></div>
  </div>
  <script>
    const form = document.getElementById('form');
    const dateInput = document.getElementById('date');
    const activityInput = document.getElementById('activity');
    const btn = document.getElementById('btn');
    const status = document.getElementById('status');

    // Default date to today
    dateInput.value = new Date().toISOString().split('T')[0];

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const date = dateInput.value;
      const activity = activityInput.value.trim();
      if (!date || !activity) return;

      btn.disabled = true;
      status.className = 'status';
      status.textContent = 'Saving...';

      try {
        const res = await fetch('/agent/invoke', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: `Save this plan: on ${date} I have ${activity}` }),
        });
        await res.json(); // consume response, don't show it
        status.textContent = '✓ Saved! Add another plan below.';
        activityInput.value = '';
        activityInput.focus();
      } catch (err) {
        status.className = 'status error';
        status.textContent = 'Something went wrong. Try again.';
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
