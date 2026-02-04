"""MCP 도구 정의."""

from fastmcp import FastMCP

from resources import info
from tools import menu, popular_restaurants


def register_resources(mcp: FastMCP) -> None:
    """MCP 서버에 리소스를 등록합니다."""

    mcp.resource("yamyam://info")(info.get_info)


def register_tools(mcp: FastMCP) -> None:
    """MCP 서버에 도구를 등록합니다."""

    mcp.tool()(menu.recommend_menu)
    mcp.tool()(popular_restaurants.get_popular_restaurants)
