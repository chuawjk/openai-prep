"""Recommender agent factory."""

from __future__ import annotations

from openai_prep.agents.factories import _create_agent, _resolve_config
from openai_prep.agents.instructions import RECOMMENDER_INSTRUCTIONS
from openai_prep.config import AppConfig
from openai_prep.schemas import RecommenderAgentSchema


def create_recommender_agent(config: AppConfig | None = None):
    """Create the recommender agent that suggests healthy activities."""
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.recommender,
        instructions=RECOMMENDER_INSTRUCTIONS,
        output_type=RecommenderAgentSchema,
    )
