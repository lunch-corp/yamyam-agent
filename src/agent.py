"""LangGraph graph entrypoint for LangGraph/LangSmith Studio.

Studio expects a LangGraph server (`langgraph dev`) at the Base URL.
This graph exposes an LLM agent that can call MCP tools.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import Field, create_model
from typing_extensions import TypedDict

from clients.mcp_client import MCPClientWrapper
from utils.config import get_model_config
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


async def _build_llm(
    model_config: dict[str, Any] | None = None, model_name: str | None = None
):
    """Create an LLM for the agent (supports OpenAI and Gemini)."""
    # YAML 파일에서 설정 로드 (파일명에서 모델 이름 추출)
    if model_config is None or model_name is None:
        model_config, model_name = await get_model_config()

    # Temperature: YAML 설정 > 기본값
    temperature = model_config.get("temperature")

    # OpenAI 모델 사용
    model_lower = model_name.lower()
    if (
        model_lower.startswith("gpt-")
        or model_lower.startswith("o1-")
        or model_lower.startswith("o3-")
    ):
        openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            raise RuntimeError("OpenAI 모델을 사용하려면 OPENAI_API_KEY 환경 변수를 설정하세요.")

        # YAML 설정에 모델 이름과 API 키 추가
        openai_config = {**model_config, "model": model_name, "api_key": openai_key}
        return ChatOpenAI(**openai_config)

    # Gemini 모델 사용
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        # 모델 이름이 지정되지 않았거나 gemini로 시작하는 경우
        if not model_name or model_name.lower().startswith("gemini-"):
            model = model_name if model_name else "gemini-2.5-flash"
            # langchain-google-genai v4+ uses google-genai underneath and accepts `client_args`.
            # We force v1 by default because some Gemini models are not available on v1beta.
            try:
                return ChatGoogleGenerativeAI(
                    model=model, temperature=temperature, google_api_key=gemini_key
                )
            except TypeError:
                # Backward-compat fallback for older langchain-google-genai which may not
                # support `client_args`.
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    google_api_key=gemini_key,
                )

    raise RuntimeError(
        "No LLM API key found. Set OPENAI_API_KEY (for OpenAI models) or "
        "GEMINI_API_KEY (for Gemini models, recommended)."
    )


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


def _build_mcp_client() -> MCPClientWrapper:
    """Create an MCP client (prefer long-running HTTP/SSE server)."""
    mcp_url = os.getenv("YAMYAM_MCP_URL")
    mcp_url_transport = os.getenv("YAMYAM_MCP_URL_TRANSPORT")  # "sse" | "streamable-http"

    if mcp_url:
        return MCPClientWrapper(url=mcp_url, url_transport=mcp_url_transport)

    # Fallback: spawn local FastMCP server via stdio. This is mainly for dev/demo.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    server_script = os.path.join(repo_root, "mcp", "server.py")
    return MCPClientWrapper(
        command=sys.executable,
        args=[server_script, "--transport", "stdio"],
        cwd=repo_root,
    )


async def _build_mcp_langchain_tools(mcp_client: MCPClientWrapper):
    """Wrap MCP tools as LangChain StructuredTool list."""

    tools = []
    for t in await mcp_client.list_tools():
        name = getattr(t, "name", None) or "unknown"
        description = getattr(t, "description", "") or ""
        input_schema = (
            getattr(t, "inputSchema", None)
            or getattr(t, "input_schema", None)
            or getattr(t, "parameters", None)
            or {}
        )
        args_schema = _json_schema_to_pydantic_model(name, input_schema)

        # 클로저 문제 해결: tool_name을 명시적으로 바인딩
        def _make_call_tool(tool_name: str):
            async def _call_tool(**kwargs: Any) -> str:
                try:
                    return str(await mcp_client.call_tool(tool_name, kwargs))
                except Exception as e:  # pragma: no cover
                    return f"MCP tool call failed: {tool_name} :: {type(e).__name__}: {e}"

            return _call_tool

        tools.append(
            StructuredTool.from_function(
                func=_make_call_tool(name),
                name=name,
                description=description,
                args_schema=args_schema,
            )
        )
    return tools


async def _get_executor():
    """Lazy init + cache a tool-calling ReAct agent graph."""
    global _executor_cache, _executor_cache_key

    # Optional: point the agent to a long-running MCP server (HTTP/SSE).
    mcp_url = os.getenv("YAMYAM_MCP_URL")
    mcp_url_transport = os.getenv("YAMYAM_MCP_URL_TRANSPORT")  # "sse" | "streamable-http"

    system_prompt = get_system_prompt()
    # Ensure compatibility with create_react_agent which typically injects tool_names too.
    if "{tool_names}" not in system_prompt:
        system_prompt = system_prompt + "\n\n도구 이름: {tool_names}\n"

    # YAML 설정을 캐시 키에 포함
    model_config, model_name = await get_model_config()
    temperature = model_config.get("temperature")

    cache_key = (mcp_url, mcp_url_transport, system_prompt, model_name, temperature)
    with _agent_lock:
        if _executor_cache is not None and _executor_cache_key == cache_key:
            return _executor_cache

        # 이미 로드한 설정을 _build_llm에 전달하여 중복 호출 방지
        llm = await _build_llm(model_config, model_name)
        mcp_client = _build_mcp_client()
        tools = await _build_mcp_langchain_tools(mcp_client)

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


async def _run(state: AgentInput) -> AgentOutput:
    query = state.get("query", "")

    executor = await _get_executor()
    result = await executor.ainvoke({"messages": [HumanMessage(content=query)]})

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
