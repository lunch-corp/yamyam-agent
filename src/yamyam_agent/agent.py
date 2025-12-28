"""Yamyam Agent 구현."""

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from .mcp_client import MCPClientSync
from .prompts import get_system_prompt
from .tool_adapter import create_langchain_tools_from_mcp

# 환경 변수 로드
load_dotenv()


class YamyamAgent:
    """Yamyam MCP Agent 클래스."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        mcp_command: str = "uv",
        mcp_args: list[str] | None = None,
        mcp_cwd: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """
        Yamyam Agent 초기화.

        Args:
            gemini_api_key: Gemini API 키 (없으면 환경 변수에서 가져옴)
            mcp_command: MCP 서버 실행 명령어
            mcp_args: MCP 서버 실행 인자
            mcp_cwd: MCP 서버 작업 디렉토리
            model_name: 사용할 Gemini 모델 이름 (기본값: gemini-2.5-flash)
        """
        # Gemini API 키 설정
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다. 환경 변수 또는 인자로 제공해주세요."
            )

        # Gemini 모델 초기화
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0,
        )

        # MCP 클라이언트 초기화 (FastMCP 서버는 표준 MCP 프로토콜을 따르므로 표준 클라이언트 사용)
        if mcp_args is None:
            # 기본값: yamyam-mcp 서버 실행
            # uv run을 사용하면 패키지가 설치된 환경에서 실행되어 모듈을 찾을 수 있음
            if mcp_command == "uv":
                mcp_args = ["run", "python", "server.py"]
            elif mcp_command == "python":
                mcp_args = ["server.py"]
            else:
                mcp_args = ["server.py"]

        # yamyam-mcp 서버가 같은 workspace에 있다고 가정
        if mcp_cwd is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # src/yamyam_agent -> yamyam-mcp
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            mcp_cwd = os.path.join(workspace_root, "yamyam-mcp")

        # PYTHONPATH에 src 디렉토리 추가 (yamyam_mcp 모듈을 찾기 위해)
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

        # MCP 도구를 LangChain 도구로 변환
        self.tools = create_langchain_tools_from_mcp(self.mcp_client)
        self.tool_map = {tool.name: tool for tool in self.tools}

        if not self.tools:
            print("⚠️  경고: MCP 서버에서 도구를 가져오지 못했습니다.")
            print(f"   MCP 서버 경로: {mcp_cwd}")
            print(f"   실행 명령어: {mcp_command} {' '.join(mcp_args)}")

        # Agent 프롬프트 생성 (system prompt 사용)
        self.prompt_template = PromptTemplate.from_template(get_system_prompt())

        # Runnable 기반 agent 체인 구성
        self.agent = self._create_agent_chain()

    def _prepare_input(self, data: dict[str, Any], tools_description: str) -> dict[str, Any]:
        """
        LLM 호출을 위한 입력 데이터를 준비합니다.

        Args:
            data: 현재 상태 데이터
            tools_description: 도구 설명 문자열

        Returns:
            프롬프트에 전달할 입력 데이터
        """
        return {
            "input": data.get("input", ""),
            "tools": tools_description,
            "agent_scratchpad": data.get("agent_scratchpad", ""),
        }

    def _call_llm(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        LLM을 호출하여 응답을 받습니다.

        Args:
            data: 프롬프트 데이터 (input, tools, agent_scratchpad 포함)

        Returns:
            LLM 응답이 추가된 데이터
        """
        prompt = self.prompt_template.format(**data)
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return {**data, "llm_response": content}

    def _parse_and_execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        LLM 응답에서 Action을 파싱하고 실행합니다.

        Args:
            data: LLM 응답이 포함된 데이터

        Returns:
            실행 결과가 포함된 데이터 (최종 답변이면 output 포함)
        """
        llm_response = data.get("llm_response", "")
        scratchpad = data.get("agent_scratchpad", "")

        # Action과 Action Input 추출
        action_match = re.search(r"Action:\s*(\w+)", llm_response)
        action_input_match = re.search(r"Action Input:\s*(.+)", llm_response, re.DOTALL)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input_str = action_input_match.group(1).strip()

            # 도구 실행
            if action in self.tool_map:
                try:
                    # Action Input을 파싱 (간단한 JSON 또는 문자열)
                    tool_input = self._parse_action_input(action_input_str)
                    tool_result = self.tool_map[action].invoke(tool_input)
                    scratchpad += f"\n{llm_response}\nObservation: {tool_result}\n"
                except Exception as e:
                    scratchpad += f"\n{llm_response}\nObservation: 에러 - {str(e)}\n"
            else:
                scratchpad += (
                    f"\n{llm_response}\nObservation: 도구 '{action}'를 찾을 수 없습니다.\n"
                )
        else:
            # 최종 답변인 경우
            return {"output": llm_response, "agent_scratchpad": scratchpad}

        return {"agent_scratchpad": scratchpad, "input": data.get("input", "")}

    def _run_with_iteration(
        self, input_data: dict[str, Any], tools_description: str, max_iterations: int = 10
    ) -> str:
        """
        Agent 실행을 반복적으로 수행합니다.

        Args:
            input_data: 초기 입력 데이터
            tools_description: 도구 설명 문자열
            max_iterations: 최대 반복 횟수

        Returns:
            최종 응답 문자열
        """
        current_data = {"input": input_data.get("input", ""), "agent_scratchpad": ""}

        for _ in range(max_iterations):
            # 입력 준비
            prepared = self._prepare_input(current_data, tools_description)
            # LLM 호출
            llm_data = self._call_llm(prepared)
            # Action 파싱 및 실행
            result = self._parse_and_execute(llm_data)

            if "output" in result:
                return result["output"]

            current_data = result

        # 최대 반복 횟수 초과 시 마지막 LLM 응답 반환
        final_prompt = self.prompt_template.format(
            input=current_data.get("input", ""),
            tools=tools_description,
            agent_scratchpad=current_data.get("agent_scratchpad", ""),
        )
        final_response = self.llm.invoke(final_prompt)
        return final_response.content if hasattr(final_response, "content") else str(final_response)

    def _create_agent_chain(self) -> Runnable:
        """Runnable 기반 agent 체인 생성."""
        tools_description = self._format_tools_description()

        def run_agent(input_data: dict[str, Any]) -> str:
            """RunnableLambda에 전달할 래퍼 함수."""
            return self._run_with_iteration(input_data, tools_description)

        return RunnableLambda(run_agent)

    def _format_tools_description(self) -> str:
        """도구 목록을 문자열로 포맷팅."""
        if not self.tools:
            return "사용 가능한 도구가 없습니다."

        descriptions = []
        for tool in self.tools:
            desc = f"- {tool.name}: {tool.description}"
            descriptions.append(desc)

        return "\n".join(descriptions)

    def _parse_action_input(self, action_input_str: str) -> dict[str, Any]:
        """Action Input 문자열을 파싱하여 딕셔너리로 변환."""
        # JSON 형식인 경우
        try:
            import json

            return json.loads(action_input_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # 단순 문자열인 경우
        # "query: hello" 같은 형식 처리
        if ":" in action_input_str:
            parts = action_input_str.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().strip("\"'")
                value = parts[1].strip().strip("\"'")
                return {key: value}

        # 기본적으로 query 키로 사용
        return {"query": action_input_str.strip().strip("\"'")}

    def run(self, query: str) -> str:
        """
        사용자 쿼리를 처리합니다.

        Args:
            query: 사용자 질문

        Returns:
            Agent 응답
        """
        try:
            result = self.agent.invoke({"input": query})
            return str(result) if result else "응답을 생성할 수 없습니다."
        except Exception as e:
            return f"에러가 발생했습니다: {str(e)}"

    def add_tool(self, tool: BaseTool) -> None:
        """
        추가 도구를 Agent에 등록합니다.

        Args:
            tool: LangChain 도구
        """
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
        # Agent 체인 재생성
        self.agent = self._create_agent_chain()

    def close(self) -> None:
        """리소스를 정리합니다."""
        self.mcp_client.close()
