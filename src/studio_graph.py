"""LangGraph graph entrypoint for LangGraph/LangSmith Studio.

Studio expects a LangGraph server (`langgraph dev`) at the Base URL.
This graph exposes an LLM agent that can call MCP tools.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import Field, create_model
from typing_extensions import TypedDict

from agents.agent import YamyamAgent
from utils.prompt_loader import get_system_prompt

_agent_lock = threading.Lock()
_executor_cache: Any | None = None
_executor_cache_key: tuple[Any, ...] | None = None


class AgentInput(TypedDict, total=False):
    """Studio input schema (only fields the user should provide)."""

    query: str


class AgentOutput(TypedDict, total=False):
    """Studio output schema (fields produced by the graph)."""

    output: str


def _build_llm():
    """Create an LLM for the agent (Gemini preferred)."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if gemini_key:
        model = os.getenv("YAMYAM_LLM_MODEL", "gemini-1.5-flash")
        temperature = float(os.getenv("YAMYAM_LLM_TEMPERATURE", "0"))
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=gemini_key,
        )

    raise RuntimeError("No LLM API key found. Set GEMINI_API_KEY (recommended).")


def _json_schema_to_pydantic_model(tool_name: str, schema: dict[str, Any]):
    """Convert MCP JSON schema to a minimal Pydantic model for StructuredTool."""

    def _to_type(s: dict[str, Any]) -> Any:
        """Best-effort JSON Schema -> Python type."""
        t = s.get("type")

        # JSON Schema allows unions, commonly like ["string", "null"].
        if isinstance(t, list):
            is_nullable = "null" in t
            base_types = [x for x in t if x != "null"]
            base_schema = {**s, "type": (base_types[0] if base_types else None)}
            inner = _to_type(base_schema)
            return (inner | None) if is_nullable else inner

        type_map: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict[str, Any],
        }

        if t == "array":
            item_schema = s.get("items") or {}
            return list[_to_type(item_schema)]  # type: ignore[misc]

        return type_map.get(t, Any)

    properties = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])

    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        prop = prop or {}
        py_type = _to_type(prop)
        is_required = name in required
        default = ... if is_required else None
        if not is_required:
            py_type = py_type | None
        fields[name] = (
            py_type,
            Field(default=default, description=prop.get("description")),
        )

    return create_model(f"MCP_{tool_name}_Args", **fields)  # type: ignore[call-arg]


def _build_mcp_langchain_tools(mcp_agent: YamyamAgent):
    """Wrap MCP tools as LangChain StructuredTool list."""

    tools = []
    for t in mcp_agent.list_tools():
        name = getattr(t, "name", None) or "unknown"
        description = getattr(t, "description", "") or ""
        input_schema = (
            getattr(t, "inputSchema", None)
            or getattr(t, "input_schema", None)
            or getattr(t, "parameters", None)
            or {}
        )
        args_schema = _json_schema_to_pydantic_model(name, input_schema)

        def _call_tool(*, _tool_name: str = name, **kwargs: Any) -> str:
            try:
                return str(mcp_agent.call_tool(_tool_name, kwargs))
            except Exception as e:  # pragma: no cover
                return f"MCP tool call failed: {_tool_name} :: {type(e).__name__}: {e}"

        tools.append(
            StructuredTool.from_function(
                func=_call_tool,
                name=name,
                description=description,
                args_schema=args_schema,
            )
        )
    return tools


def _get_executor():
    """Lazy init + cache a tool-calling ReAct agent graph."""
    global _executor_cache, _executor_cache_key

    # Optional: point the agent to a long-running MCP server (HTTP/SSE).
    mcp_url = os.getenv("YAMYAM_MCP_URL")
    mcp_url_transport = os.getenv("YAMYAM_MCP_URL_TRANSPORT")  # "sse" | "streamable-http"

    system_prompt = get_system_prompt()
    # Ensure compatibility with create_react_agent which typically injects tool_names too.
    if "{tool_names}" not in system_prompt:
        system_prompt = system_prompt + "\n\n도구 이름: {tool_names}\n"

    model_name = os.getenv("YAMYAM_LLM_MODEL")
    temperature = os.getenv("YAMYAM_LLM_TEMPERATURE")

    cache_key = (mcp_url, mcp_url_transport, system_prompt, model_name, temperature)
    with _agent_lock:
        if _executor_cache is not None and _executor_cache_key == cache_key:
            return _executor_cache

        llm = _build_llm()
        mcp_agent = YamyamAgent(mcp_url=mcp_url, mcp_url_transport=mcp_url_transport)
        tools = _build_mcp_langchain_tools(mcp_agent)

        tool_names = [getattr(t, "name", "unknown") for t in tools]
        tool_names_str = ", ".join(tool_names)
        if "{tool_names}" in system_prompt:
            system_prompt_rendered = system_prompt.replace("{tool_names}", tool_names_str)
        else:
            system_prompt_rendered = system_prompt + f"\n\n도구 이름: {tool_names_str}\n"

        _executor_cache = create_agent(
            llm,
            tools,
            system_prompt=system_prompt_rendered,
        )
        _executor_cache_key = cache_key
        return _executor_cache


def _run(state: AgentInput) -> AgentOutput:
    query = state.get("query", "")

    executor = _get_executor()
    result = executor.invoke({"messages": [HumanMessage(content=query)]})

    if isinstance(result, dict) and result.get("messages"):
        last = result["messages"][-1]
        content = getattr(last, "content", None)
        output = str(content) if content is not None else str(last)
        return {"output": output or ""}

    return {"output": str(result) or ""}


_builder: StateGraph = StateGraph(AgentInput, input_schema=AgentInput, output_schema=AgentOutput)
_builder.add_node("yamyam_agent", _run)
_builder.set_entry_point("yamyam_agent")
_builder.add_edge("yamyam_agent", END)

graph = _builder.compile()
