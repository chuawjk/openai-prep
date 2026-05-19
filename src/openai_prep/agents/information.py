"""Information agent factory."""

from __future__ import annotations

from openai_prep.agents.factories import _create_agent, _resolve_config
from openai_prep.agents.instructions import INFORMATION_INSTRUCTIONS
from openai_prep.config import AppConfig
from openai_prep.schemas import InformationAgentSchema


def create_information_agent(config: AppConfig | None = None):
    """Create the information agent that searches HealthHub and returns cited advice."""
    # Deferred import avoids loading the agents SDK WebSearchTool on package import.
    from openai_prep.tools import create_healthhub_search_tool

    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.information,
        instructions=INFORMATION_INSTRUCTIONS,
        tools=[create_healthhub_search_tool(resolved_config.search)],
        output_type=InformationAgentSchema,
    )
