"""LangGraph graph entrypoint for LangGraph/LangSmith Studio.

Studio expects a LangGraph server (`langgraph dev`) at the Base URL.
This graph exposes an LLM agent that can call MCP tools.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import Field, create_model
from typing_extensions import TypedDict

from clients.mcp_client import MCPClientWrapper
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
    """Create an LLM for the agent (OpenAI only)."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OpenAI 모델을 사용하려면 OPENAI_API_KEY 환경 변수를 설정하세요.")

    # YAML 설정에 모델 이름과 API 키 추가
    openai_config = {"model": "gpt-4o-mini", "api_key": openai_key}
    return ChatOpenAI(**openai_config)


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

        # 동기 함수로 감싸서 에이전트가 await 안 해도 도구 결과가 문자열로 들어가게 함
        def _make_call_tool(tool_name: str):
            async def _call_async(**kwargs: Any) -> str:
                try:
                    return str(await mcp_client.call_tool(tool_name, kwargs))
                except Exception as e:  # pragma: no cover
                    return f"MCP tool call failed: {tool_name} :: {type(e).__name__}: {e}"

            def _call_sync(**kwargs: Any) -> str:
                try:
                    return asyncio.run(_call_async(**kwargs))
                except RuntimeError as e:
                    if "cannot be called from a running event loop" in str(e):
                        loop = asyncio.new_event_loop()
                        try:
                            return loop.run_until_complete(_call_async(**kwargs))
                        finally:
                            loop.close()
                    raise
                except Exception as e:  # pragma: no cover
                    return f"MCP tool call failed: {tool_name} :: {type(e).__name__}: {e}"

            return _call_sync

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

    cache_key = (mcp_url, mcp_url_transport, system_prompt)
    with _agent_lock:
        if _executor_cache is not None and _executor_cache_key == cache_key:
            return _executor_cache

        llm = _build_llm()
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


def _message_content_to_str(m: Any) -> str:
    """메시지 content를 문자열로 추출 (다양한 형식 지원)."""
    raw = getattr(m, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(
                    item.get("text") or item.get("content") or str(item)
                )
            else:
                parts.append(str(item))
        return " ".join(str(p).strip() for p in parts if p).strip()
    return str(raw).strip()


def _last_tool_error(messages: list) -> str | None:
    """마지막 도구 호출 결과 중 오류로 보이는 내용을 반환. 없으면 None."""
    err_keywords = (
        "오류",
        "연결 실패",
        "연결 오류",
        "API 오류",
        "API 연결",
        "MCP tool call failed",
        "failed",
        "Error",
        "timeout",
        "Timeout",
        "YAMYAM_OPS_API_URL",
    )
    # 1) ToolMessage / type=="tool" 메시지에서 오류 찾기
    for m in reversed(messages):
        name = getattr(m, "type", None) or type(m).__name__
        is_tool = (
            name == "tool"
            or name == "ToolMessage"
            or "tool" in str(name).lower()
            or isinstance(m, ToolMessage)
        )
        if not is_tool:
            continue
        s = _message_content_to_str(m)
        if s and any(k in s for k in err_keywords):
            return s
    # 2) 위에서 못 찾으면 마지막 제외한 메시지에서 오류 문구 검사 (도구 결과가 다른 타입일 수 있음)
    for m in list(reversed(messages))[1:]:
        s = _message_content_to_str(m)
        if s and any(k in s for k in err_keywords):
            return s
    return None


async def _run(state: AgentInput) -> AgentOutput:
    query = state.get("query", "")

    executor = await _get_executor()
    result = await executor.ainvoke({"messages": [HumanMessage(content=query)]})

    if isinstance(result, dict) and result.get("messages"):
        messages = result["messages"]
        last = messages[-1]
        output = _message_content_to_str(last) or str(last)

        # 도구 오류가 있으면 맨 앞에 노출
        tool_err = _last_tool_error(messages)
        if tool_err and tool_err not in output:
            output = "**[오류]**\n" + tool_err + "\n\n---\n" + (output or "")
        elif output and ("문제가 발생" in output or "다시 시도" in output):
            # 마지막 AI 직전 메시지 = 도구 결과일 가능성 높음 → 그대로 맨 앞에 붙임
            if len(messages) >= 2:
                prev = _message_content_to_str(messages[-2])
                if (
                    prev
                    and prev not in output
                    and "coroutine object" not in prev
                ):
                    output = "**[도구 반환 내용]**\n" + prev + "\n\n---\n" + output
            if "**[도구 반환 내용]**" not in output:
                output = (
                    "**[도구 결과를 파싱하지 못함]** 아래가 AI 답변입니다. "
                    "실제 오류는 MCP/도구 로그에서 확인하세요.\n\n---\n"
                    + output
                )

        return {"output": output or ""}

    return {"output": str(result) or ""}


_builder: StateGraph = StateGraph(AgentInput, input_schema=AgentInput, output_schema=AgentOutput)
_builder.add_node("yamyam_agent", _run)
_builder.set_entry_point("yamyam_agent")
_builder.add_edge("yamyam_agent", END)

graph = _builder.compile()
