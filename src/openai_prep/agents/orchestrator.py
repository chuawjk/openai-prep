"""Orchestrator agent factory."""

from __future__ import annotations

from openai_prep.agents.factories import _create_agent, _resolve_config
from openai_prep.agents.instructions import ORCHESTRATOR_INSTRUCTIONS
from openai_prep.config import AppConfig
from openai_prep.schemas import OrchestratorAgentSchema


def create_orchestrator_agent(config: AppConfig | None = None):
    """Create the orchestrator agent that classifies incoming requests."""
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.orchestrator,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        output_type=OrchestratorAgentSchema,
    )
