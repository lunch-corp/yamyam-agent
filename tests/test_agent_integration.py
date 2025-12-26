"""Agent 통합 테스트 (실제 MCP 서버 연결 필요)."""

import os

import pytest
from dotenv import load_dotenv

from yamyam_agent.agent import YamyamAgent

load_dotenv()


@pytest.mark.integration
def test_agent_initialization():
    """Agent가 정상적으로 초기화되는지 테스트."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY가 설정되지 않았습니다.")

    agent = YamyamAgent(gemini_api_key=api_key)
    assert agent is not None
    assert agent.llm is not None
    assert agent.tools is not None
    assert len(agent.tools) > 0

    agent.close()


@pytest.mark.integration
def test_agent_without_api_key():
    """API 키 없이 Agent 초기화 시 에러가 발생하는지 테스트."""
    # 환경 변수 백업
    original_key = os.environ.get("GEMINI_API_KEY")
    try:
        # 환경 변수 제거
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            YamyamAgent()
    finally:
        # 환경 변수 복원
        if original_key:
            os.environ["GEMINI_API_KEY"] = original_key


@pytest.mark.integration
def test_agent_run_simple_query():
    """Agent가 간단한 쿼리를 처리할 수 있는지 테스트."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY가 설정되지 않았습니다.")

    agent = YamyamAgent(gemini_api_key=api_key)

    try:
        response = agent.run("안녕하세요!")
        assert response is not None
        assert len(response) > 0
    finally:
        agent.close()
