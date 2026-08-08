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
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Schedule Planner</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #eef2ff;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px 16px;
    }
    .card {
      background: #fff;
      border-radius: 18px;
      padding: 36px 32px;
      width: 100%;
      max-width: 480px;
      box-shadow: 0 10px 40px rgba(79,110,247,0.12);
    }
    h1 { font-size: 1.5rem; color: #1a1a2e; margin-bottom: 4px; }
    .subtitle { color: #888; font-size: 0.88rem; margin-bottom: 28px; }

    /* Tab toggle */
    .tabs { display: flex; gap: 8px; margin-bottom: 24px; }
    .tab {
      flex: 1; padding: 10px; border: 2px solid #dde1f0;
      border-radius: 10px; background: #fff; font-size: 0.9rem;
      font-weight: 600; color: #888; cursor: pointer; transition: all 0.18s;
    }
    .tab.active { border-color: #4f6ef7; background: #4f6ef7; color: #fff; }

    /* Panels */
    .panel { display: none; }
    .panel.active { display: block; }

    label { display: block; font-size: 0.82rem; font-weight: 600; color: #555; margin-bottom: 5px; margin-top: 16px; }
    label:first-of-type { margin-top: 0; }
    input[type="date"], input[type="text"], textarea {
      width: 100%; padding: 11px 13px;
      border: 1.5px solid #dde1f0; border-radius: 8px;
      font-size: 0.97rem; color: #1a1a2e; outline: none;
      transition: border-color 0.18s; font-family: inherit;
    }
    input:focus, textarea:focus { border-color: #4f6ef7; }
    textarea { resize: vertical; min-height: 80px; }

    .btn {
      margin-top: 20px; width: 100%; padding: 12px;
      background: #4f6ef7; color: #fff; border: none;
      border-radius: 9px; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: background 0.18s;
    }
    .btn:hover:not(:disabled) { background: #3a57e8; }
    .btn:disabled { background: #b0bcf7; cursor: not-allowed; }

    .status { margin-top: 14px; font-size: 0.88rem; min-height: 20px; color: #4f6ef7; text-align: center; }
    .status.error { color: #e53e3e; }

    /* Answer box */
    .answer {
      margin-top: 18px; background: #f5f7ff;
      border: 1.5px solid #dde1f0; border-radius: 10px;
      padding: 14px 16px; font-size: 0.93rem; color: #1a1a2e;
      line-height: 1.6; display: none; white-space: pre-wrap;
    }
  </style>
</head>
<body>
<div class="card">
  <h1>📅 Schedule Planner</h1>
  <p class="subtitle">Save your plans or ask about your schedule</p>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('save', this)">💾 Save Plan</button>
    <button class="tab" onclick="switchTab('ask', this)">🔍 Ask Schedule</button>
  </div>

  <!-- SAVE PANEL -->
  <div class="panel active" id="panel-save">
    <label for="date">Date</label>
    <input type="date" id="date" required/>
    <label for="purpose">What are you doing?</label>
    <input type="text" id="purpose" placeholder="e.g. Team offsite, Dentist at 3pm" autocomplete="off"/>
    <button class="btn" id="save-btn" onclick="savePlan()">Save Plan</button>
    <div class="status" id="save-status"></div>
  </div>

  <!-- ASK PANEL -->
  <div class="panel" id="panel-ask">
    <label for="question">Your question</label>
    <textarea id="question" placeholder="e.g. Do I have anything on Aug 20? Can I go to a movie on Saturday?"></textarea>
    <button class="btn" id="ask-btn" onclick="askAgent()">Ask</button>
    <div class="answer" id="answer"></div>
    <div class="status error" id="ask-status"></div>
  </div>
</div>

<script>
  // Default date to today
  document.getElementById('date').value = new Date().toISOString().split('T')[0];

  function switchTab(name, el) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('panel-' + name).classList.add('active');
  }

  async function savePlan() {
    const date = document.getElementById('date').value;
    const purpose = document.getElementById('purpose').value.trim();
    const btn = document.getElementById('save-btn');
    const status = document.getElementById('save-status');
    if (!date || !purpose) { status.className='status error'; status.textContent='Please fill in both fields.'; return; }
    btn.disabled = true;
    status.className = 'status';
    status.textContent = 'Saving...';
    try {
      const res = await fetch('/agent/invoke', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ input: `Save this plan: on ${date} I have ${purpose}` }),
      });
      await res.json();
      status.textContent = '✓ Plan saved! Add another below.';
      document.getElementById('purpose').value = '';
      document.getElementById('purpose').focus();
    } catch {
      status.className = 'status error';
      status.textContent = 'Something went wrong. Try again.';
    } finally { btn.disabled = false; }
  }

  async function askAgent() {
    const q = document.getElementById('question').value.trim();
    const btn = document.getElementById('ask-btn');
    const answer = document.getElementById('answer');
    const status = document.getElementById('ask-status');
    if (!q) return;
    btn.disabled = true;
    answer.style.display = 'none';
    status.textContent = '';
    btn.textContent = 'Thinking...';
    try {
      const res = await fetch('/agent/invoke', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ input: q }),
      });
      const data = await res.json();
      answer.style.display = 'block';
      answer.textContent = data.output || 'No response.';
    } catch {
      status.textContent = 'Something went wrong. Try again.';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Ask';
    }
  }

  // Allow Enter in question textarea with Ctrl+Enter
  document.getElementById('question').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) askAgent();
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
