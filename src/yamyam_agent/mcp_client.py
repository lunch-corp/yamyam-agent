"""MCP 클라이언트 래퍼."""

import asyncio
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

except ImportError:
    # mcp 패키지가 없을 경우를 대비한 fallback
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


class MCPClientWrapper:
    """MCP 서버와 통신하는 클라이언트 래퍼."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        MCP 클라이언트 초기화.

        Args:
            command: MCP 서버 실행 명령어
            args: MCP 서버 실행 인자
            cwd: 작업 디렉토리
            env: 환경 변수 딕셔너리
        """
        if StdioServerParameters is None:
            raise ImportError(
                "mcp 패키지가 설치되지 않았습니다. 'uv sync' 또는 'pip install mcp'를 실행하세요."
            )

        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.server_params = StdioServerParameters(
            command=command,
            args=self.args,
            env=env,
            cwd=cwd,
        )
        # 각 호출마다 새로운 연결을 생성하므로 세션을 저장하지 않음

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        MCP 서버에서 사용 가능한 도구 목록을 가져옵니다.

        Returns:
            도구 목록
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools if hasattr(result, "tools") else []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """
        MCP 도구를 호출합니다.

        Args:
            name: 도구 이름
            arguments: 도구 인자

        Returns:
            도구 실행 결과
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments or {})
                return result.content[0].text if result.content else None

    async def close(self) -> None:
        """MCP 서버 연결을 종료합니다."""
        # 연결은 각 호출마다 새로 생성되므로 여기서는 아무것도 하지 않음
        pass


class MCPClientSync:
    """동기식 MCP 클라이언트 래퍼."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        MCP 클라이언트 초기화.

        Args:
            command: MCP 서버 실행 명령어 (예: "uv")
            args: MCP 서버 실행 인자 (예: ["run", "yamyam-mcp"])
            cwd: 작업 디렉토리
            env: 환경 변수 딕셔너리
        """
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.client = MCPClientWrapper(command, args, cwd, env)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """이벤트 루프를 가져오거나 생성합니다."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def list_tools(self) -> list[dict[str, Any]]:
        """동기식으로 도구 목록을 가져옵니다."""
        loop = self._ensure_loop()
        return loop.run_until_complete(self.client.list_tools())

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """동기식으로 도구를 호출합니다."""
        loop = self._ensure_loop()
        return loop.run_until_complete(self.client.call_tool(name, arguments))

    def close(self) -> None:
        """연결을 종료합니다."""
        loop = self._ensure_loop()
        loop.run_until_complete(self.client.close())
