"""Rejection agent factory."""

from __future__ import annotations

from openai_prep.agents.factories import _create_agent, _resolve_config
from openai_prep.agents.instructions import REJECTION_INSTRUCTIONS
from openai_prep.config import AppConfig


def create_rejection_agent(config: AppConfig | None = None):
    """Create the rejection agent that politely declines out-of-scope requests."""
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.rejection,
        instructions=REJECTION_INSTRUCTIONS,
    )
