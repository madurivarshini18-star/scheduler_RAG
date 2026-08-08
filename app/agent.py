from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, schedule_maker

TOOLS = [schedule_maker, get_schedule]

llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def _system_prompt() -> str:
    """Rebuilt on every call so 'today' is always current on a long-running
    server, not frozen at import time."""
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return (
        f"You are a scheduling assistant. Today's date is {today}.\n\n"
        "Rules:\n"
        "1. Always resolve relative dates ('tomorrow', 'next Friday', "
        "'this weekend') into an exact YYYY-MM-DD date before calling any "
        "tool. Use today's date above as the reference point.\n"
        "2. Use `schedule_maker` whenever the person tells you about a plan, "
        "event, or commitment to save.\n"
        "3. Use `get_schedule` whenever the person asks what they're doing, "
        "whether they're free, or whether something conflicts with an "
        "existing plan on a given date.\n"
        "4. When answering a conflict-check question (e.g. 'can I go to a "
        "movie on that day'), call `get_schedule` first, then reason about "
        "whether the existing plans conflict, and explain your answer "
        "clearly and briefly.\n"
        "5. If the person gives a date range, you may call `get_schedule` "
        "once per date in the range.\n"
        "6. Never invent schedule entries — only report what the tool "
        "returns."
    )


# Cache the compiled graph per calendar day so we rebuild at most once per day
# (keeps "today" accurate across midnight without rebuilding every request).
_graph_cache: dict[str, object] = {}


def _get_graph():
    day_key = datetime.now().strftime("%Y-%m-%d")
    if day_key not in _graph_cache:
        _graph_cache.clear()  # drop yesterday's cached graph
        _graph_cache[day_key] = create_react_agent(
            model=llm,
            tools=TOOLS,
            state_modifier=SystemMessage(content=_system_prompt()),
        )
    return _graph_cache[day_key]


def run_agent(user_input: str) -> str:
    graph = _get_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    last_message = result["messages"][-1]
    return last_message.content
