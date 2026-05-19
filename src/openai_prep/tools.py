"""Tool factories for OpenAI Prep agents."""

from __future__ import annotations

from agents import WebSearchTool

from openai_prep.config import SearchConfig


def create_healthhub_search_tool(search_config: SearchConfig | None = None) -> WebSearchTool:
    """Create the prototype-equivalent HealthHub-restricted web search tool."""

    resolved_config = search_config or SearchConfig()
    return WebSearchTool(
        filters=resolved_config.filters(),
        search_context_size=resolved_config.search_context_size,
        user_location=resolved_config.user_location_dict(),
    )
