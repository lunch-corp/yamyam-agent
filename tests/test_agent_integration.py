"""Agent 통합 테스트 (실제 MCP 서버 연결 필요)."""

import pytest
from dotenv import load_dotenv

from yamyam_agent.agent import YamyamAgent

load_dotenv()


@pytest.mark.integration
def test_agent_initialization():
    """Agent가 정상적으로 초기화되는지 테스트."""
    agent = YamyamAgent()
    assert agent is not None
    tools = agent.list_tools()
    assert tools is not None
    assert len(tools) > 0

    agent.close()


@pytest.mark.integration
def test_agent_run_simple_query():
    """Agent가 간단한 쿼리를 처리할 수 있는지 테스트."""
    agent = YamyamAgent()

    try:
        response = agent.run("안녕하세요!")
        assert response is not None
        assert len(response) > 0
    finally:
        agent.close()
