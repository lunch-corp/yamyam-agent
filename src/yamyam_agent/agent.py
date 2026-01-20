"""Yamyam Agent 구현.

현재는 “추론(LLM)” 대신, 연결된 MCP 서버에 등록된 도구 중 하나를 골라
그 정보를 출력하는 최소 기능만 제공합니다.
"""

import os
import sys
from typing import Any

from dotenv import load_dotenv

from .mcp_client import MCPClientSync

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
        if mcp_url:
            # 이미 실행 중인 MCP 서버로 연결 (SSE/Streamable HTTP)
            self.mcp_client = MCPClientSync(
                url=mcp_url,
                url_transport=mcp_url_transport,
            )
        else:
            # 로컬 stdio 서버를 프로세스로 실행 (기본)
            # 기본적으로는 현재 파이썬 인터프리터로 서버를 띄워,
            # uv 설치/캐시 이슈에 덜 민감하게 합니다.
            if mcp_command is None:
                mcp_command = sys.executable

            if mcp_args is None:
                # 기본값: 이 레포의 `mcp/server.py` 실행
                # uv run을 사용하면 venv/lock 환경에서 실행되어 의존성을 안정적으로 찾을 수 있음
                if os.path.basename(mcp_command) == "uv":
                    mcp_args = ["run", "python", "mcp/server.py"]
                else:
                    # python 계열 인터프리터라면 -u(버퍼링 비활성)로 stdio 프로토콜 안정성 향상
                    mcp_args = ["-u", "mcp/server.py"]

            # MCP 서버는 이 workspace 내부(`mcp/`)에서 관리한다고 가정
            if mcp_cwd is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # src/yamyam_agent -> src -> (project root)
                project_root = os.path.dirname(os.path.dirname(current_dir))
                mcp_cwd = project_root

            # PYTHONPATH에 src 디렉토리 추가 (`src/` 레이아웃 패키지를 찾기 위해)
            mcp_src_path = os.path.join(mcp_cwd, "src")
            env = os.environ.copy()
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{mcp_src_path}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = mcp_src_path

            self.mcp_client = MCPClientSync(
                command=mcp_command,
                args=mcp_args,
                cwd=mcp_cwd,
                env=env,
            )

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

    def run(self, _query: str = "") -> str:
        """현재 등록된 MCP 도구 중 하나를 보여줍니다."""
        return self.describe_one_tool(self.tool_name)

    def close(self) -> None:
        """리소스를 정리합니다."""
        self.mcp_client.close()
