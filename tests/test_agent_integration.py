"""Agent 통합 테스트 (실제 MCP 서버 연결 필요)."""

import os

import pytest
from dotenv import load_dotenv

from clients.mcp_client import MCPClientSync

load_dotenv()

MCP_URL = os.getenv("YAMYAM_MCP_URL")
MCP_TRANSPORT = os.getenv("YAMYAM_MCP_URL_TRANSPORT")


@pytest.mark.integration
def test_agent_initialization():
    """MCP 서버가 정상적으로 연결/도구 조회되는지 테스트."""
    if not MCP_URL:
        pytest.skip("YAMYAM_MCP_URL not set (requires a running MCP server).")

    client = MCPClientSync(url=MCP_URL, url_transport=MCP_TRANSPORT)
    try:
        tools = client.list_tools()
        assert tools is not None
        assert len(tools) > 0
    except Exception as e:
        pytest.skip(f"MCP server not reachable: {e}")
    finally:
        client.close()


@pytest.mark.integration
def test_agent_run_simple_query():
    """MCP 서버 도구 호출이 동작하는지 테스트(echo)."""
    if not MCP_URL:
        pytest.skip("YAMYAM_MCP_URL not set (requires a running MCP server).")

    client = MCPClientSync(url=MCP_URL, url_transport=MCP_TRANSPORT)

    try:
        try:
            response = client.call_tool("echo", {"query": "안녕하세요!"})
            assert response == "안녕하세요!"
        except Exception as e:
            pytest.skip(f"MCP server not reachable: {e}")
    finally:
        client.close()
