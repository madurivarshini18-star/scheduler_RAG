import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
