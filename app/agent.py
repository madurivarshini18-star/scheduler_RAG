import json
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, update_schedule

# Map tool names to callables
TOOLS = {
    "get_schedule": get_schedule,
    "update_schedule": update_schedule,
}

llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are an intelligent scheduling assistant. Today is {today}.

You have two tools:
- get_schedule: Retrieve schedule entries by date or natural language query.
- update_schedule: Add, update, or delete schedule entries.

Rules:
1. Resolve relative dates (tomorrow, next Friday, this weekend) to exact YYYY-MM-DD using today as reference.
2. For any question about what is scheduled, call get_schedule first then answer based on results.
3. For adding/moving/removing events, call update_schedule with action "add", "update", or "delete".
4. For availability checks, call get_schedule for that date and report findings.
5. Never invent events — only report what the tools return.
6. Confirm all changes clearly to the user."""


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """Parse XML-style tool calls that Groq sometimes generates instead of JSON."""
    calls = []
    pattern = r'<function=(\w+)(\{.*?\})</function>'
    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            calls.append({"name": name, "args": args, "id": f"call_{len(calls)}"})
        except json.JSONDecodeError:
            pass
    return calls


def run_agent(user_input: str) -> str:
    messages = [
        SystemMessage(content=_system_prompt()),
        HumanMessage(content=user_input),
    ]

    for _ in range(5):  # max 5 tool-call rounds
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Get tool calls — either from structured response or XML fallback
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
        elif hasattr(response, "content") and isinstance(response.content, str):
            tool_calls = _parse_xml_tool_calls(response.content)

        if not tool_calls:
            # No tool calls — return final answer
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            return content or "I'm sorry, I couldn't process that request."

        # Execute each tool call
        for call in tool_calls:
            name = call.get("name") or call.get("function", {}).get("name", "")
            args = call.get("args") or call.get("function", {}).get("arguments", {})
            call_id = call.get("id", "call_0")

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            tool_fn = TOOLS.get(name)
            if tool_fn:
                try:
                    result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    return "I reached the maximum number of steps. Please try a more specific question."
