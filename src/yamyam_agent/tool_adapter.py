"""MCP 도구를 LangChain 도구로 변환하는 어댑터."""

from typing import Any

try:
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool
from pydantic import Field


class MCPToolAdapter(BaseTool):
    """MCP 도구를 LangChain 도구로 변환하는 어댑터."""

    mcp_client: Any = Field(description="MCP 클라이언트 인스턴스")
    tool_name: str = Field(description="MCP 도구 이름")
    tool_description: str = Field(description="MCP 도구 설명")
    tool_parameters: dict[str, Any] = Field(
        default_factory=dict, description="MCP 도구 파라미터 스키마"
    )

    name: str = ""
    description: str = ""

    def __init__(self, mcp_client: Any, tool_info: Any, **kwargs):
        """
        MCP 도구 어댑터 초기화.

        Args:
            mcp_client: MCP 클라이언트 인스턴스
            tool_info: MCP 도구 정보 (dict 또는 Tool 객체)
            **kwargs: 추가 인자
        """
        # Tool 객체인지 딕셔너리인지 확인
        if hasattr(tool_info, "name"):
            # Tool 객체인 경우
            tool_name = tool_info.name
            tool_description = getattr(tool_info, "description", "")
            # inputSchema는 다양한 속성명으로 올 수 있음
            input_schema = {}
            if hasattr(tool_info, "inputSchema"):
                input_schema = tool_info.inputSchema
            elif hasattr(tool_info, "input_schema"):
                input_schema = tool_info.input_schema
            elif hasattr(tool_info, "parameters"):
                input_schema = tool_info.parameters
        else:
            # 딕셔너리인 경우
            tool_name = tool_info.get("name", "")
            tool_description = tool_info.get("description", "")
            input_schema = tool_info.get(
                "inputSchema",
                tool_info.get("input_schema", tool_info.get("parameters", {})),
            )

        super().__init__(
            name=tool_name,
            description=tool_description,
            mcp_client=mcp_client,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=input_schema,
            **kwargs,
        )

    def _run(self, **kwargs: Any) -> str:
        """
        도구를 실행합니다.

        Args:
            **kwargs: 도구 인자

        Returns:
            실행 결과 문자열
        """
        try:
            result = self.mcp_client.call_tool(self.tool_name, kwargs)
            if isinstance(result, str):
                return result
            return str(result)
        except Exception as e:
            return f"Error executing tool {self.tool_name}: {str(e)}"

    async def _arun(self, **kwargs: Any) -> str:
        """
        비동기적으로 도구를 실행합니다.

        Args:
            **kwargs: 도구 인자

        Returns:
            실행 결과 문자열
        """
        # 동기 클라이언트를 사용하므로 동일한 방식으로 처리
        return self._run(**kwargs)


def create_langchain_tools_from_mcp(mcp_client: Any) -> list[BaseTool]:
    """
    MCP 클라이언트의 도구 목록을 LangChain 도구로 변환합니다.

    Args:
        mcp_client: MCP 클라이언트 인스턴스 (list_tools 메서드 제공)

    Returns:
        LangChain 도구 리스트
    """
    tools = []
    try:
        print("MCP 서버에서 도구 목록을 가져오는 중...")
        mcp_tools = mcp_client.list_tools()
        print(f"MCP 서버에서 {len(mcp_tools)}개의 도구를 찾았습니다.")

        for tool_info in mcp_tools:
            try:
                # Tool 객체인지 딕셔너리인지 확인
                if hasattr(tool_info, "name"):
                    tool_name = tool_info.name
                else:
                    tool_name = tool_info.get("name", "unknown")
                print(f"  - 도구 등록: {tool_name}")
                adapter = MCPToolAdapter(mcp_client=mcp_client, tool_info=tool_info)
                tools.append(adapter)
            except Exception as e:
                if hasattr(tool_info, "name"):
                    tool_name = tool_info.name
                else:
                    tool_name = tool_info.get("name", "unknown")
                print(f"  ⚠️  도구 '{tool_name}' 등록 실패: {e}")
                import traceback

                traceback.print_exc()
    except Exception as e:
        print(f"❌ MCP 서버에서 도구를 가져오는 중 에러 발생: {e}")
        import traceback

        traceback.print_exc()
    return tools
