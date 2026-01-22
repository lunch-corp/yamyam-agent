"""MCP 클라이언트 래퍼."""

import asyncio
from typing import Any

from fastmcp.client.client import Client
from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport


class MCPClientWrapper:
    """MCP 서버와 통신하는 클라이언트 래퍼."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        url_transport: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        MCP 클라이언트 초기화.

        Args:
            command: MCP 서버 실행 명령어 (stdio 실행 시)
            args: MCP 서버 실행 인자
            url: 이미 실행 중인 MCP 서버 URL (SSE/Streamable HTTP)
            url_transport: url 사용 시 전송 방식 ("sse" | "streamable-http")
            cwd: 작업 디렉토리
            env: 환경 변수 딕셔너리
        """
        if not url and not command:
            raise ValueError("Either 'url' or 'command' must be provided.")

        self.command = command
        self.args = args or []
        self.url = url
        self.url_transport = url_transport
        self.cwd = cwd
        self.env = env
        # 컨텍스트 매니저를 미리 생성하여 재사용합니다.
        self.client = Client(self._client_target())

    def _client_target(self) -> Any:
        if self.url:
            if self.url_transport == "sse":
                return SSETransport(self.url)
            if self.url_transport in ("streamable-http", "http"):
                return StreamableHttpTransport(self.url)
            # default: FastMCP settings default is streamable-http path `/mcp`,
            # but most users will pass an explicit url like http://host:port/sse.
            if str(self.url).endswith("/sse"):
                return SSETransport(self.url)
            return StreamableHttpTransport(self.url)

        return StdioTransport(
            command=self.command or "python",
            args=self.args,
            env=self.env,
            cwd=self.cwd,
            keep_alive=False,
        )

    async def list_tools(self) -> list[Any]:
        """
        MCP 서버에서 사용 가능한 도구 목록을 가져옵니다.

        Returns:
            도구 목록
        """
        try:
            async with self.client as client:
                return await client.list_tools()
        except Exception as e:  # pragma: no cover - message enrichment
            raise RuntimeError(f"{self._format_target()} :: {type(e).__name__}: {e!r}") from e

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """
        MCP 도구를 호출합니다.

        Args:
            name: 도구 이름
            arguments: 도구 인자

        Returns:
            도구 실행 결과
        """
        try:
            async with self.client as client:
                result = await client.call_tool(name, arguments or {})
                content = getattr(result, "content", None)
                if not content:
                    return None
                first = content[0]
                return getattr(first, "text", str(first))
        except Exception as e:  # pragma: no cover - message enrichment
            raise RuntimeError(
                f"{self._format_target()} tool={name} :: {type(e).__name__}: {e!r}"
            ) from e

    def _format_target(self) -> str:
        if self.url:
            transport = self.url_transport or "auto"
            return f"MCP client failed to connect (url={self.url} transport={transport})"
        return (
            "MCP client failed to connect ("
            f"command={self.command!r} args={self.args!r} cwd={self.cwd!r}"
            ")"
        )

    async def close(self) -> None:
        """MCP 서버 연결을 종료합니다."""
        # 연결은 각 호출마다 새로 생성/종료되므로 여기서는 아무것도 하지 않음
        pass


class MCPClientSync:
    """동기식 MCP 클라이언트 래퍼."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        url_transport: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        MCP 클라이언트 초기화.

        Args:
            command: MCP 서버 실행 명령어 (예: "uv")
            args: MCP 서버 실행 인자 (예: ["run", "yamyam-mcp"])
            url: 이미 실행 중인 MCP 서버 URL (SSE/Streamable HTTP)
            url_transport: url 사용 시 전송 방식 ("sse" | "streamable-http")
            cwd: 작업 디렉토리
            env: 환경 변수 딕셔너리
        """
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.url = url
        self.url_transport = url_transport
        self.client = MCPClientWrapper(
            command=command,
            args=args,
            url=url,
            url_transport=url_transport,
            cwd=cwd,
            env=env,
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """이벤트 루프를 가져오거나 생성합니다."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def list_tools(self) -> list[Any]:
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
