"""Synthesis agent factory."""

from __future__ import annotations

from openai_prep.agents.factories import _create_agent, _resolve_config
from openai_prep.agents.instructions import SYNTHESIS_INSTRUCTIONS
from openai_prep.config import AppConfig


def create_synthesis_agent(config: AppConfig | None = None):
    """Create the synthesis agent that combines downstream agent outputs."""
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.synthesis,
        instructions=SYNTHESIS_INSTRUCTIONS,
    )
