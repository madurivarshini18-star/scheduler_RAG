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
body{font-family:'Segoe UI',sans-serif;background:#f0f2ff;min-height:100vh;color:#1a1a2e}
.app{display:grid;grid-template-columns:220px 1fr;min-height:100vh}

/* SIDEBAR */
.sidebar{background:#1e2147;padding:24px 0;display:flex;flex-direction:column}
.logo{padding:0 20px 24px;font-size:1.1rem;font-weight:700;color:#fff;border-bottom:1px solid #2d3170}
.logo span{font-size:1.4rem;margin-right:8px}
.nav-section{padding:16px 12px 4px;font-size:.7rem;font-weight:700;color:#6b7db3;letter-spacing:.08em;text-transform:uppercase}
.nav-btn{display:flex;align-items:center;gap:10px;width:100%;padding:10px 16px;background:none;border:none;color:#b0bce8;font-size:.88rem;cursor:pointer;border-radius:8px;margin:2px 8px;width:calc(100% - 16px);text-align:left;transition:all .15s}
.nav-btn:hover{background:#2d3170;color:#fff}
.nav-btn.active{background:#4f6ef7;color:#fff}
.nav-btn .icon{font-size:1.1rem;width:20px;text-align:center}
.cat-btn{display:flex;align-items:center;gap:10px;width:calc(100% - 16px);padding:8px 16px;background:none;border:none;color:#b0bce8;font-size:.83rem;cursor:pointer;border-radius:8px;margin:2px 8px;text-align:left;transition:all .15s}
.cat-btn:hover{background:#2d3170;color:#fff}
.cat-btn.active{background:rgba(79,110,247,.3);color:#7b9fff}
.cat-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

/* MAIN */
.main{display:flex;flex-direction:column;overflow:hidden}
.topbar{background:#fff;padding:16px 28px;border-bottom:1px solid #e8eaf6;display:flex;align-items:center;justify-content:space-between}
.topbar h1{font-size:1.15rem;color:#1a1a2e}
.topbar .date{font-size:.83rem;color:#888}
.content{padding:24px 28px;overflow-y:auto;flex:1}

/* CARDS */
.section-title{font-size:.8rem;font-weight:700;color:#888;letter-spacing:.06em;text-transform:uppercase;margin-bottom:12px;margin-top:20px}
.section-title:first-child{margin-top:0}
.event-grid{display:flex;flex-direction:column;gap:8px}
.event-card{background:#fff;border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);transition:box-shadow .15s}
.event-card:hover{box-shadow:0 3px 12px rgba(0,0,0,.1)}
.event-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.event-info{flex:1}
.event-title{font-weight:600;font-size:.93rem;color:#1a1a2e}
.event-meta{font-size:.78rem;color:#888;margin-top:2px}
.event-badge{font-size:.7rem;padding:3px 8px;border-radius:20px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.del-btn{background:none;border:none;color:#ccc;cursor:pointer;font-size:1rem;padding:4px 6px;border-radius:6px;transition:all .15s}
.del-btn:hover{color:#e53e3e;background:#fff0f0}
.empty{text-align:center;padding:48px 0;color:#aaa;font-size:.9rem}

/* FORM PANEL */
.form-panel{background:#fff;border-radius:14px;padding:24px;box-shadow:0 2px 12px rgba(0,0,0,.07);max-width:520px}
.form-panel h2{font-size:1rem;font-weight:700;margin-bottom:20px;color:#1a1a2e}
label{display:block;font-size:.78rem;font-weight:600;color:#555;margin-bottom:4px;margin-top:14px}
label:first-of-type{margin-top:0}
input,select,textarea{width:100%;padding:10px 12px;border:1.5px solid #e0e4f0;border-radius:8px;font-size:.9rem;color:#1a1a2e;outline:none;font-family:inherit;transition:border-color .15s}
input:focus,select:focus,textarea:focus{border-color:#4f6ef7}
textarea{min-height:64px;resize:vertical}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn-primary{margin-top:18px;width:100%;padding:11px;background:#4f6ef7;color:#fff;border:none;border-radius:9px;font-size:.93rem;font-weight:600;cursor:pointer;transition:background .15s}
.btn-primary:hover:not(:disabled){background:#3a57e8}
.btn-primary:disabled{background:#b0bcf7;cursor:not-allowed}
.form-status{margin-top:10px;font-size:.82rem;text-align:center;min-height:16px;color:#4f6ef7}
.form-status.err{color:#e53e3e}

/* ASK PANEL */
.ask-panel{background:#fff;border-radius:14px;padding:24px;box-shadow:0 2px 12px rgba(0,0,0,.07);max-width:560px}
.ask-panel h2{font-size:1rem;font-weight:700;margin-bottom:16px;color:#1a1a2e}
.ask-answer{margin-top:14px;background:#f5f7ff;border:1.5px solid #dde1f0;border-radius:10px;padding:14px;font-size:.88rem;line-height:1.7;display:none;white-space:pre-wrap;color:#1a1a2e}
.ask-err{margin-top:8px;font-size:.82rem;color:#e53e3e;min-height:14px}
</style>
</head>
<body>
<div class="app">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="logo"><span>📅</span>Scheduler</div>

  <div class="nav-section">Views</div>
  <button class="nav-btn active" onclick="showView('all')" id="nav-all">
    <span class="icon">🗓️</span>All Events
  </button>
  <button class="nav-btn" onclick="showView('add')" id="nav-add">
    <span class="icon">➕</span>Add Event
  </button>
  <button class="nav-btn" onclick="showView('ask')" id="nav-ask">
    <span class="icon">🔍</span>Ask Schedule
  </button>

  <div class="nav-section" style="margin-top:12px">Categories</div>
  <button class="cat-btn active" onclick="filterCat('all')" id="cat-all">
    <span class="cat-dot" style="background:#4f6ef7"></span>All
  </button>
  <button class="cat-btn" onclick="filterCat('meeting')" id="cat-meeting">
    <span class="cat-dot" style="background:#0ea5e9"></span>Meeting
  </button>
  <button class="cat-btn" onclick="filterCat('appointment')" id="cat-appointment">
    <span class="cat-dot" style="background:#8b5cf6"></span>Appointment
  </button>
  <button class="cat-btn" onclick="filterCat('task')" id="cat-task">
    <span class="cat-dot" style="background:#f59e0b"></span>Task
  </button>
  <button class="cat-btn" onclick="filterCat('workshop')" id="cat-workshop">
    <span class="cat-dot" style="background:#10b981"></span>Workshop
  </button>
  <button class="cat-btn" onclick="filterCat('office')" id="cat-office">
    <span class="cat-dot" style="background:#3b82f6"></span>Office
  </button>
  <button class="cat-btn" onclick="filterCat('trip')" id="cat-trip">
    <span class="cat-dot" style="background:#ef4444"></span>Trip
  </button>
  <button class="cat-btn" onclick="filterCat('shopping')" id="cat-shopping">
    <span class="cat-dot" style="background:#ec4899"></span>Shopping
  </button>
  <button class="cat-btn" onclick="filterCat('going_out')" id="cat-going_out">
    <span class="cat-dot" style="background:#f97316"></span>Going Out
  </button>
  <button class="cat-btn" onclick="filterCat('party')" id="cat-party">
    <span class="cat-dot" style="background:#a855f7"></span>Party
  </button>
  <button class="cat-btn" onclick="filterCat('gym')" id="cat-gym">
    <span class="cat-dot" style="background:#14b8a6"></span>Gym
  </button>
  <button class="cat-btn" onclick="filterCat('health')" id="cat-health">
    <span class="cat-dot" style="background:#22c55e"></span>Health
  </button>
  <button class="cat-btn" onclick="filterCat('personal')" id="cat-personal">
    <span class="cat-dot" style="background:#6366f1"></span>Personal
  </button>
</div>

<!-- MAIN CONTENT -->
<div class="main">
  <div class="topbar">
    <h1 id="view-title">All Scheduled Events</h1>
    <span class="date" id="today-date"></span>
  </div>
  <div class="content">

    <!-- ALL EVENTS VIEW -->
    <div id="view-all">
      <div id="events-container"><div class="empty">Loading your schedule...</div></div>
    </div>

    <!-- ADD EVENT VIEW -->
    <div id="view-add" style="display:none">
      <div class="form-panel">
        <h2>Add / Update / Delete Event</h2>
        <label>Action</label>
        <select id="f-action">
          <option value="add">Add new event</option>
          <option value="update">Update event</option>
          <option value="delete">Delete event</option>
        </select>
        <div class="row2">
          <div><label>Date</label><input type="date" id="f-date"/></div>
          <div><label>Time</label><input type="time" id="f-time" value="09:00"/></div>
        </div>
        <label>Event Title</label>
        <input type="text" id="f-title" placeholder="e.g. Team standup, Grocery run"/>
        <label>Category</label>
        <select id="f-type">
          <option value="meeting">Meeting</option>
          <option value="appointment">Appointment</option>
          <option value="task">Task</option>
          <option value="workshop">Workshop</option>
          <option value="office">Office</option>
          <option value="trip">Trip</option>
          <option value="shopping">Shopping</option>
          <option value="going_out">Going Out</option>
          <option value="party">Party</option>
          <option value="gym">Gym</option>
          <option value="health">Health</option>
          <option value="personal">Personal</option>
        </select>
        <label>Description (optional)</label>
        <input type="text" id="f-desc" placeholder="Extra details..."/>
        <button class="btn-primary" id="save-btn" onclick="saveEvent()">Save</button>
        <div class="form-status" id="save-status"></div>
      </div>
    </div>

    <!-- ASK VIEW -->
    <div id="view-ask" style="display:none">
      <div class="ask-panel">
        <h2>Ask about your schedule</h2>
        <label>Your question</label>
        <textarea id="q-input" placeholder="e.g. What do I have tomorrow?&#10;Am I free Friday afternoon?&#10;Move my 2 PM meeting to 4 PM"></textarea>
        <button class="btn-primary" id="ask-btn" onclick="askAgent()">Ask</button>
        <div class="ask-answer" id="ask-answer"></div>
        <div class="ask-err" id="ask-err"></div>
      </div>
    </div>

  </div>
</div>
</div>

<script>
const CAT_COLORS = {
  meeting:'#0ea5e9', appointment:'#8b5cf6', task:'#f59e0b',
  workshop:'#10b981', office:'#3b82f6', trip:'#ef4444',
  shopping:'#ec4899', going_out:'#f97316', party:'#a855f7',
  gym:'#14b8a6', health:'#22c55e', personal:'#6366f1', event:'#4f6ef7'
};
const CAT_BG = {
  meeting:'#e0f2fe', appointment:'#ede9fe', task:'#fef3c7',
  workshop:'#d1fae5', office:'#dbeafe', trip:'#fee2e2',
  shopping:'#fce7f3', going_out:'#ffedd5', party:'#f3e8ff',
  gym:'#ccfbf1', health:'#dcfce7', personal:'#e0e7ff', event:'#eef2ff'
};

let allEvents = [];
let currentCat = 'all';

// Set today's date
const today = new Date();
document.getElementById('today-date').textContent = today.toLocaleDateString('en-US',{weekday:'long',year:'numeric',month:'long',day:'numeric'});
document.getElementById('f-date').value = today.toISOString().split('T')[0];

async function invoke(input) {
  const res = await fetch('/agent/invoke', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({input})
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data.output;
}

// ── VIEW SWITCHING ──
function showView(name) {
  ['all','add','ask'].forEach(v => {
    document.getElementById('view-'+v).style.display = v===name ? '' : 'none';
    document.getElementById('nav-'+v).classList.toggle('active', v===name);
  });
  const titles = {all:'All Scheduled Events', add:'Add / Update Event', ask:'Ask Schedule Assistant'};
  document.getElementById('view-title').textContent = titles[name];
  if (name==='all') loadEvents();
}

// ── CATEGORY FILTER ──
function filterCat(cat) {
  currentCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('cat-'+cat).classList.add('active');
  showView('all');
  renderEvents();
}

// ── LOAD EVENTS ──
async function loadEvents() {
  document.getElementById('events-container').innerHTML = '<div class="empty">Loading...</div>';
  try {
    const output = await invoke('Show me all scheduled events for the next 30 days');
    allEvents = parseEvents(output);
    renderEvents();
  } catch(e) {
    document.getElementById('events-container').innerHTML = '<div class="empty">Could not load events: '+e.message+'</div>';
  }
}

function parseEvents(text) {
  const events = [];
  const lines = text.split('\\n');
  for (const line of lines) {
    const m = line.match(/\\[([A-Z_]+)\\]\\s*(\\d{4}-\\d{2}-\\d{2})\\s+(\\d{2}:\\d{2})\\s*-\\s*(.+?)(?:\\s*\\|\\s*(.*))?$/);
    if (m) {
      events.push({
        type: m[1].toLowerCase(),
        date: m[2],
        time: m[3],
        title: m[4].trim(),
        desc: (m[5]||'').trim()
      });
    }
  }
  return events;
}

function renderEvents() {
  const container = document.getElementById('events-container');
  let filtered = currentCat==='all' ? allEvents : allEvents.filter(e=>e.type===currentCat);

  if (!filtered.length) {
    container.innerHTML = '<div class="empty">No events found' + (currentCat!=='all' ? ' in this category' : '') + '.</div>';
    return;
  }

  // Group by date
  const grouped = {};
  filtered.forEach(e => {
    if (!grouped[e.date]) grouped[e.date] = [];
    grouped[e.date].push(e);
  });

  const sortedDates = Object.keys(grouped).sort();
  let html = '';
  sortedDates.forEach(date => {
    const label = formatDateLabel(date);
    html += `<div class="section-title">${label}</div><div class="event-grid">`;
    grouped[date].forEach(e => {
      const color = CAT_COLORS[e.type] || CAT_COLORS.event;
      const bg = CAT_BG[e.type] || CAT_BG.event;
      html += `<div class="event-card">
        <span class="event-dot" style="background:${color}"></span>
        <div class="event-info">
          <div class="event-title">${e.title}</div>
          <div class="event-meta">${e.time}${e.desc ? ' · '+e.desc : ''}</div>
        </div>
        <span class="event-badge" style="background:${bg};color:${color}">${e.type.replace('_',' ')}</span>
        <button class="del-btn" title="Delete" onclick="deleteEvent('${e.date}','${e.title.replace(/'/g,"\\\\'")}')">🗑️</button>
      </div>`;
    });
    html += '</div>';
  });
  container.innerHTML = html;
}

function formatDateLabel(dateStr) {
  const d = new Date(dateStr + 'T00:00:00');
  const todayStr = today.toISOString().split('T')[0];
  const tomorrowStr = new Date(today.getTime()+86400000).toISOString().split('T')[0];
  if (dateStr === todayStr) return 'Today — ' + d.toLocaleDateString('en-US',{weekday:'long',month:'short',day:'numeric'});
  if (dateStr === tomorrowStr) return 'Tomorrow — ' + d.toLocaleDateString('en-US',{weekday:'long',month:'short',day:'numeric'});
  return d.toLocaleDateString('en-US',{weekday:'long',year:'numeric',month:'long',day:'numeric'});
}

// ── DELETE ──
async function deleteEvent(date, title) {
  if (!confirm(`Delete "${title}" on ${date}?`)) return;
  try {
    await invoke(`Delete the event titled "${title}" on ${date}`);
    allEvents = allEvents.filter(e => !(e.date===date && e.title===title));
    renderEvents();
  } catch(e) { alert('Delete failed: ' + e.message); }
}

// ── SAVE EVENT ──
async function saveEvent() {
  const action = document.getElementById('f-action').value;
  const date   = document.getElementById('f-date').value;
  const time   = document.getElementById('f-time').value || '09:00';
  const title  = document.getElementById('f-title').value.trim();
  const type   = document.getElementById('f-type').value;
  const desc   = document.getElementById('f-desc').value.trim();
  const btn    = document.getElementById('save-btn');
  const status = document.getElementById('save-status');

  if (!date || !title) { status.className='form-status err'; status.textContent='Date and title are required.'; return; }
  btn.disabled=true; status.className='form-status'; status.textContent='Saving...';

  const prompt = action==='delete'
    ? `Delete the event titled "${title}" on ${date}`
    : action==='update'
    ? `Update the event titled "${title}" on ${date}, change time to ${time}. Type: ${type}. Description: ${desc}`
    : `Add a ${type} titled "${title}" on ${date} at ${time}. ${desc}`;

  try {
    await invoke(prompt);
    status.textContent = action==='delete' ? '✓ Deleted.' : action==='update' ? '✓ Updated.' : '✓ Event saved!';
    if (action==='add') { document.getElementById('f-title').value=''; document.getElementById('f-desc').value=''; }
  } catch(e) {
    status.className='form-status err'; status.textContent='Error: '+e.message;
  } finally { btn.disabled=false; }
}

// ── ASK AGENT ──
async function askAgent() {
  const q   = document.getElementById('q-input').value.trim();
  const btn = document.getElementById('ask-btn');
  const ans = document.getElementById('ask-answer');
  const err = document.getElementById('ask-err');
  if (!q) return;
  btn.disabled=true; btn.textContent='Thinking...';
  ans.style.display='none'; err.textContent='';
  try {
    const output = await invoke(q);
    ans.style.display='block'; ans.textContent=output;
  } catch(e) { err.textContent='Error: '+e.message; }
  finally { btn.disabled=false; btn.textContent='Ask'; }
}

document.getElementById('q-input').addEventListener('keydown', e=>{
  if(e.key==='Enter'&&e.ctrlKey) askAgent();
});

// Initial load
loadEvents();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


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
