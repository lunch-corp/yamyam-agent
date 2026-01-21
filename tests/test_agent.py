"""간단한 MCP 클라이언트 테스트."""

import os
import sys
from pathlib import Path

from clients.mcp_client import MCPClientSync


def test_mcp_stdio_echo() -> None:
    """로컬 FastMCP 서버(stdio)로 echo 호출이 되는지 테스트."""
    repo_root = Path(__file__).resolve().parents[1]
    server_script = repo_root / "mcp" / "server.py"

    client = MCPClientSync(
        command=sys.executable,
        args=[str(server_script), "--transport", "stdio"],
        cwd=str(repo_root),
        env={k: v for k, v in os.environ.items() if isinstance(v, str)},
    )

    tools = client.list_tools()
    tool_names = {getattr(t, "name", "") for t in tools}
    assert "echo" in tool_names

    out = client.call_tool("echo", {"query": "Hello, World!"})
    assert out == "Hello, World!"

    client.close()
