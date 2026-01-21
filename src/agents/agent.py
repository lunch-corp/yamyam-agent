"""Yamyam Agent 구현.

현재는 “추론(LLM)” 대신, 연결된 MCP 서버에 등록된 도구 중 하나를 골라
그 정보를 출력하는 최소 기능만 제공합니다.
"""

from typing import Any

from dotenv import load_dotenv

from clients.mcp_client import MCPClientSync

load_dotenv()


class YamyamAgent:
    """Yamyam MCP Agent 클래스."""

    def __init__(
        self,
        # Backward-compat: previously used for LLM-based agent.
        gemini_api_key: str | None = None,
        mcp_url: str | None = None,
        mcp_url_transport: str | None = None,
        mcp_command: str | None = None,
        mcp_args: list[str] | None = None,
        mcp_cwd: str | None = None,
        # Backward-compat: previously selected Gemini model.
        model_name: str | None = None,
        tool_name: str | None = None,
    ):
        """
        Yamyam Agent 초기화.

        Args:
            gemini_api_key: (호환성) 예전 LLM 모드에서 사용. 현재는 사용하지 않음.
            mcp_url: 이미 실행 중인 MCP 서버 URL (예: "http://127.0.0.1:8000/sse" 또는 ".../mcp")
            mcp_url_transport: URL 사용 시 전송 방식 ("sse" | "streamable-http")
            mcp_command: MCP 서버 실행 명령어
            mcp_args: MCP 서버 실행 인자
            mcp_cwd: MCP 서버 작업 디렉토리
            model_name: (호환성) 예전 LLM 모드에서 사용. 현재는 사용하지 않음.
            tool_name: 출력할 MCP 도구 이름 (없으면 첫 번째 도구)
        """
        self.tool_name = tool_name
        self._unused_gemini_api_key = gemini_api_key
        self._unused_model_name = model_name

        # MCP 클라이언트 초기화
        try:
            # 이미 실행 중인 MCP 서버로 연결 (SSE/Streamable HTTP)
            self.mcp_client = MCPClientSync(
                url=mcp_url,
                url_transport=mcp_url_transport,
            )
        except Exception as e:
            raise RuntimeError(f"MCP 서버 초기화 실패: {type(e).__name__}: {e}") from e

        self._cached_tools: list[Any] | None = None

    def list_tools(self) -> list[Any]:
        """MCP 서버에서 도구 목록을 가져옵니다(캐시)."""
        if self._cached_tools is None:
            self._cached_tools = self.mcp_client.list_tools()
        return self._cached_tools

    def describe_one_tool(self, tool_name: str | None = None) -> str:
        """등록된 MCP 도구 중 하나를 선택해 정보를 문자열로 반환합니다."""
        tools = self.list_tools()
        if not tools:
            return "MCP 서버에 등록된 도구가 없습니다."

        selected = None
        if tool_name:
            for t in tools:
                if getattr(t, "name", None) == tool_name:
                    selected = t
                    break
        if selected is None:
            selected = tools[0]

        name = getattr(selected, "name", "unknown")
        desc = getattr(selected, "description", "") or ""
        schema = (
            getattr(selected, "inputSchema", None)
            or getattr(selected, "input_schema", None)
            or getattr(selected, "parameters", None)
            or {}
        )

        return f"선택된 MCP 도구\n- name: {name}\n- description: {desc}\n- inputSchema: {schema}\n"

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """MCP 도구를 호출합니다."""
        return self.mcp_client.call_tool(name, arguments or {})

    def _default_arguments_for_tool(self, tool_name: str, query: str) -> dict[str, Any]:
        """간단한 데모용: query를 각 도구의 기본 입력 필드에 매핑합니다."""
        if tool_name == "recommend_menu":
            return {"request": query}
        if tool_name == "echo":
            return {"query": query}
        return {}

    def _tool_exists(self, name: str) -> bool:
        return any(getattr(t, "name", None) == name for t in self.list_tools())

    def run(self, query: str = "") -> str:
        """간단 실행 엔트리포인트.

        - tool_name이 지정되면: 해당 MCP 도구를 호출(쿼리 없으면 설명 출력)
        - tool_name이 없으면: query에 따라 기본 동작 수행
          - "역할/뭐 할 수" 질문: 소개 + 도구 목록
          - 추천/먹을거 질문: recommend_menu가 있으면 호출
          - 그 외: echo가 있으면 echo 호출
        """
        if not query and not self.tool_name:
            return self.describe_one_tool(None)

        if not self.tool_name:
            q = (query or "").strip()
            q_l = q.lower()
            if q and any(
                k in q_l for k in ["역할", "뭐", "뭘", "할 수", "할수", "기능", "help", "사용법"]
            ):
                tool_names = [getattr(t, "name", "unknown") for t in self.list_tools()]
                return (
                    "나는 MCP 서버에 등록된 도구를 찾아 호출하는 간단한 에이전트야.\n"
                    f"사용 가능한 도구: {', '.join(tool_names) if tool_names else '(없음)'}\n"
                    "특정 도구를 쓰려면 tool_name을 지정하거나, "
                    "추천/메뉴 요청이면 recommend_menu를 자동 호출해."
                )

            if q and any(
                k in q_l for k in ["추천", "메뉴", "뭐 먹", "뭘 먹", "먹을", "점심", "저녁"]
            ):
                if self._tool_exists("recommend_menu"):
                    try:
                        return str(self.call_tool("recommend_menu", {"request": q}))
                    except Exception as e:
                        return f"도구 호출 실패: recommend_menu :: {type(e).__name__}: {e}"

            if q and self._tool_exists("echo"):
                try:
                    return str(self.call_tool("echo", {"query": q}))
                except Exception as e:
                    return f"도구 호출 실패: echo :: {type(e).__name__}: {e}"

            return self.describe_one_tool(None)

        # tool_name이 지정된 경우
        if not query:
            return self.describe_one_tool(self.tool_name)

        if not self._tool_exists(self.tool_name):
            return f"도구를 찾을 수 없습니다: {self.tool_name}\n\n" + self.describe_one_tool(None)

        args = self._default_arguments_for_tool(self.tool_name, query)
        try:
            result = self.call_tool(self.tool_name, args)
            return str(result)
        except Exception as e:
            return f"도구 호출 실패: {self.tool_name} :: {type(e).__name__}: {e}"

    def close(self) -> None:
        """리소스를 정리합니다."""
        self.mcp_client.close()
