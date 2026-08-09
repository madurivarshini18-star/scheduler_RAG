from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, update_schedule

TOOLS = [get_schedule, update_schedule]

# Bind tools explicitly so Groq receives proper JSON tool schemas
llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

_graph_cache: dict[str, object] = {}


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are an intelligent scheduling assistant. Today is {today}.

You have two tools:
- get_schedule: Retrieve schedule entries by date or natural language query.
- update_schedule: Add, update, or delete schedule entries.

Rules:
1. Resolve relative dates (tomorrow, next Friday, this weekend) to exact YYYY-MM-DD dates using today as reference.
2. For any question about what is scheduled, always call get_schedule first.
3. For adding/moving/removing events, call update_schedule with action "add", "update", or "delete".
4. For availability checks ("Am I free on X?"), call get_schedule for that date and report what you find.
5. Never invent events — only report what the tools return.
6. Always confirm update/delete actions clearly to the user."""


def _get_graph():
    day_key = datetime.now().strftime("%Y-%m-%d")
    if day_key not in _graph_cache:
        _graph_cache.clear()
        _graph_cache[day_key] = create_react_agent(
            model=llm_with_tools,
            tools=TOOLS,
            state_modifier=SystemMessage(content=_system_prompt()),
        )
    return _graph_cache[day_key]


def run_agent(user_input: str) -> str:
    graph = _get_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content
