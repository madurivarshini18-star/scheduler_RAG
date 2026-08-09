import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.seed import seed
        seed()
    except Exception as e:
        print(f"Seed skipped: {e}")
    yield


from app.agent import run_agent

app = FastAPI(
    title="Schedule RAG Agent",
    version="2.0",
    description="Agentic scheduling assistant — 30-day schedule with RAG",
    lifespan=lifespan,
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
    try:
        return AgentOutput(output=run_agent(payload.input))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/stream")
def agent_stream(payload: AgentInput) -> StreamingResponse:
    def _generate():
        output = run_agent(payload.input)
        yield f"data: {json.dumps({'output': output})}\n\n"
    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agent/playground", response_class=HTMLResponse)
def playground():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Schedule Assistant</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#eef2ff;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px 16px}
.card{background:#fff;border-radius:18px;padding:36px 32px;width:100%;max-width:520px;box-shadow:0 10px 40px rgba(79,110,247,.13)}
h1{font-size:1.5rem;color:#1a1a2e;margin-bottom:4px}
.subtitle{color:#888;font-size:.88rem;margin-bottom:24px}
.tabs{display:flex;gap:8px;margin-bottom:24px}
.tab{flex:1;padding:10px;border:2px solid #dde1f0;border-radius:10px;background:#fff;font-size:.9rem;font-weight:600;color:#888;cursor:pointer;transition:all .18s}
.tab.active{border-color:#4f6ef7;background:#4f6ef7;color:#fff}
.panel{display:none}.panel.active{display:block}
label{display:block;font-size:.82rem;font-weight:600;color:#555;margin-bottom:5px;margin-top:16px}
label:first-of-type{margin-top:0}
input[type=date],input[type=text],input[type=time],select,textarea{width:100%;padding:11px 13px;border:1.5px solid #dde1f0;border-radius:8px;font-size:.97rem;color:#1a1a2e;outline:none;transition:border-color .18s;font-family:inherit}
input:focus,select:focus,textarea:focus{border-color:#4f6ef7}
textarea{resize:vertical;min-height:72px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{margin-top:20px;width:100%;padding:12px;background:#4f6ef7;color:#fff;border:none;border-radius:9px;font-size:1rem;font-weight:600;cursor:pointer;transition:background .18s}
.btn:hover:not(:disabled){background:#3a57e8}
.btn:disabled{background:#b0bcf7;cursor:not-allowed}
.status{margin-top:12px;font-size:.88rem;min-height:18px;color:#4f6ef7;text-align:center}
.status.error{color:#e53e3e}
.answer{margin-top:16px;background:#f5f7ff;border:1.5px solid #dde1f0;border-radius:10px;padding:14px 16px;font-size:.93rem;color:#1a1a2e;line-height:1.7;display:none;white-space:pre-wrap}
</style>
</head>
<body>
<div class="card">
  <h1>📅 Schedule Assistant</h1>
  <p class="subtitle">Manage your 30-day schedule or ask anything about it</p>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('save',this)">💾 Save / Update</button>
    <button class="tab" onclick="switchTab('ask',this)">🔍 Ask Schedule</button>
  </div>

  <!-- SAVE PANEL -->
  <div class="panel active" id="panel-save">
    <label>Action</label>
    <select id="action">
      <option value="add">Add new event</option>
      <option value="update">Update existing event</option>
      <option value="delete">Delete event</option>
    </select>
    <div class="row">
      <div><label>Date</label><input type="date" id="date"/></div>
      <div><label>Time</label><input type="time" id="time" value="09:00"/></div>
    </div>
    <label>Event Title</label>
    <input type="text" id="title" placeholder="e.g. Team standup, Dentist appointment"/>
    <label>Event Type</label>
    <select id="etype">
      <option value="meeting">Meeting</option>
      <option value="appointment">Appointment</option>
      <option value="task">Task</option>
      <option value="workshop">Workshop</option>
    </select>
    <label>Description (optional)</label>
    <input type="text" id="desc" placeholder="Extra details..."/>
    <button class="btn" id="save-btn" onclick="savePlan()">Save</button>
    <div class="status" id="save-status"></div>
  </div>

  <!-- ASK PANEL -->
  <div class="panel" id="panel-ask">
    <label>Your question</label>
    <textarea id="question" placeholder="e.g. What do I have tomorrow?&#10;Am I free Friday afternoon?&#10;Move my 2 PM meeting to 4 PM"></textarea>
    <button class="btn" id="ask-btn" onclick="askAgent()">Ask</button>
    <div class="answer" id="answer"></div>
    <div class="status error" id="ask-status"></div>
  </div>
</div>
<script>
document.getElementById('date').value = new Date().toISOString().split('T')[0];

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-'+name).classList.add('active');
}

async function savePlan() {
  const action = document.getElementById('action').value;
  const date   = document.getElementById('date').value;
  const time   = document.getElementById('time').value || '09:00';
  const title  = document.getElementById('title').value.trim();
  const etype  = document.getElementById('etype').value;
  const desc   = document.getElementById('desc').value.trim();
  const btn    = document.getElementById('save-btn');
  const status = document.getElementById('save-status');

  if (!date || !title) {
    status.className='status error';
    status.textContent='Date and title are required.';
    return;
  }
  btn.disabled=true;
  status.className='status';
  status.textContent='Processing...';

  const prompt = action==='delete'
    ? `Delete the event titled "${title}" on ${date}`
    : action==='update'
    ? `Update the event titled "${title}" on ${date} — change time to ${time}. Description: ${desc}`
    : `Add a ${etype} titled "${title}" on ${date} at ${time}. ${desc}`;

  try {
    const res = await fetch('/agent/invoke',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input: prompt}),
    });
    await res.json();
    status.textContent = action==='delete' ? '✓ Event deleted.' : action==='update' ? '✓ Event updated.' : '✓ Event saved!';
    if (action==='add') {
      document.getElementById('title').value='';
      document.getElementById('desc').value='';
      document.getElementById('title').focus();
    }
  } catch {
    status.className='status error';
    status.textContent='Something went wrong. Try again.';
  } finally { btn.disabled=false; }
}

async function askAgent() {
  const q      = document.getElementById('question').value.trim();
  const btn    = document.getElementById('ask-btn');
  const answer = document.getElementById('answer');
  const status = document.getElementById('ask-status');
  if (!q) return;
  btn.disabled=true; btn.textContent='Thinking...';
  answer.style.display='none'; status.textContent='';
  try {
    const res = await fetch('/agent/invoke',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input:q}),
    });
    const data = await res.json();
    if (!res.ok) {
      status.textContent = 'Error: ' + (data.detail || res.statusText);
    } else {
      answer.style.display='block';
      answer.textContent = data.output || 'No response.';
    }
  } catch(err) {
    status.textContent='Network error: ' + err.message;
  } finally { btn.disabled=false; btn.textContent='Ask'; }
}

document.getElementById('question').addEventListener('keydown', e=>{
  if(e.key==='Enter' && e.ctrlKey) askAgent();
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
