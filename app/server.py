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
    """Simple chat UI to interact with the scheduling agent."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Schedule Agent</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; background: #f9f9f9; }
    h1 { font-size: 1.4rem; margin-bottom: 4px; }
    p  { color: #555; margin-top: 0; font-size: 0.9rem; }
    #chat { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 300px; max-height: 500px; overflow-y: auto; margin-bottom: 12px; }
    .msg { margin: 8px 0; line-height: 1.5; }
    .user   { color: #1a56db; font-weight: 600; }
    .agent  { color: #111; }
    .thinking { color: #aaa; font-style: italic; }
    #form { display: flex; gap: 8px; }
    #input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; }
    button { padding: 10px 20px; background: #1a56db; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 1rem; }
    button:disabled { background: #aaa; cursor: not-allowed; }
  </style>
</head>
<body>
  <h1>📅 Schedule Agent</h1>
  <p>Tell me your plans or ask what you have on a given day.</p>
  <div id="chat"></div>
  <form id="form">
    <input id="input" type="text" placeholder="e.g. I have a meeting on 2026-08-20" autocomplete="off" />
    <button id="btn" type="submit">Send</button>
  </form>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const btn = document.getElementById('btn');

    function addMsg(text, cls) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      btn.disabled = true;
      addMsg('You: ' + text, 'user');
      const thinking = addMsg('Agent is thinking...', 'thinking');
      try {
        const res = await fetch('/agent/invoke', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: text }),
        });
        const data = await res.json();
        thinking.remove();
        addMsg('Agent: ' + (data.output || data.detail || JSON.stringify(data)), 'agent');
      } catch (err) {
        thinking.remove();
        addMsg('Error: ' + err.message, 'thinking');
      } finally {
        btn.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
