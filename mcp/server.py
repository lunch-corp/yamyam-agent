import argparse

from fastmcp import FastMCP
from mcp_tools import resources, tools

# FastMCP 서버 인스턴스 생성
mcp = FastMCP("yamyam-mcp")

# 도구 및 리소스 등록
tools.register_tools(mcp)
resources.register_resources(mcp)


def main() -> None:
    """MCP 서버를 실행합니다."""

    parser = argparse.ArgumentParser(description="Yamyam MCP Server (FastMCP)")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports")
    # 8000 is commonly used by langgraph dev / other local servers.
    parser.add_argument("--port", type=int, default=8001, help="Port for HTTP transports")
    parser.add_argument(
        "--path",
        default=None,
        help="Endpoint path for HTTP transports (default: FastMCP settings, sse=/sse, http=/mcp)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    # For HTTP transports, run an ASGI server (uvicorn) under the hood.
    mcp.run(transport=args.transport, host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
