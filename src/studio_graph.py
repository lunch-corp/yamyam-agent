"""LangGraph graph entrypoint for LangGraph/LangSmith Studio.

Studio expects a LangGraph server (`langgraph dev`) at the Base URL.
This graph wraps `YamyamAgent.run()` so you can drive runs from Studio.
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.agent import YamyamAgent  # noqa: E402


class AgentState(TypedDict, total=False):
    """Minimal state schema for Studio."""
    query: str


def _run(state: AgentState) -> AgentState:
    query = state.get("query", "")

    # Optional: point the agent to a long-running MCP server (HTTP/SSE).
    mcp_url = os.getenv("YAMYAM_MCP_URL")
    mcp_url_transport = os.getenv("YAMYAM_MCP_URL_TRANSPORT")  # "sse" | "streamable-http"
    tool_name = os.getenv("YAMYAM_TOOL_NAME")

    agent = YamyamAgent(
        mcp_url=mcp_url,
        mcp_url_transport=mcp_url_transport,
        tool_name=tool_name,
    )
    try:
        output = agent.run(query)
        return {"output": output}
    finally:
        agent.close()


_builder: StateGraph = StateGraph(AgentState)
_builder.add_node("yamyam_agent", _run)
_builder.set_entry_point("yamyam_agent")
_builder.add_edge("yamyam_agent", END)

graph = _builder.compile()
