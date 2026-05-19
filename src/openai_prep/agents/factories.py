"""Shared agent factory helpers and multi-agent factory for the OpenAI Prep workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai_prep.config import AgentConfig, AppConfig, ModelSettingsConfig, default_config

if TYPE_CHECKING:
    from agents import Agent, ModelSettings, Tool


@dataclass(frozen=True, slots=True)
class AgentFactories:
    """Container for all workflow agents."""

    orchestrator: Agent
    recommender: Agent
    information: Agent
    synthesis: Agent
    rejection: Agent


def create_agents(config: AppConfig | None = None) -> AgentFactories:
    """Create and return all workflow agents."""
    # Deferred imports break the circular dependency between factories.py and
    # the per-agent modules, and prevent loading the SDK on package import.
    from openai_prep.agents.information import create_information_agent
    from openai_prep.agents.orchestrator import create_orchestrator_agent
    from openai_prep.agents.recommender import create_recommender_agent
    from openai_prep.agents.rejection import create_rejection_agent
    from openai_prep.agents.synthesis import create_synthesis_agent

    resolved_config = _resolve_config(config)
    return AgentFactories(
        orchestrator=create_orchestrator_agent(resolved_config),
        recommender=create_recommender_agent(resolved_config),
        information=create_information_agent(resolved_config),
        synthesis=create_synthesis_agent(resolved_config),
        rejection=create_rejection_agent(resolved_config),
    )


def _create_agent(
    agent_config: AgentConfig,
    *,
    instructions: str,
    output_type: type | None = None,
    tools: list[Tool] | None = None,
) -> Agent:
    # Deferred import prevents loading 749 SDK modules when openai_prep.agents is
    # first imported; the SDK is only initialised when a factory function is called.
    from agents import Agent, ModelSettings  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "name": agent_config.name,
        "instructions": instructions,
        "model": agent_config.model,
        "model_settings": _create_model_settings(agent_config.model_settings),
    }
    if output_type is not None:
        kwargs["output_type"] = output_type
    if tools is not None:
        kwargs["tools"] = tools
    return Agent(**kwargs)


def _create_model_settings(model_settings: ModelSettingsConfig) -> ModelSettings:
    # Deferred import prevents loading 749 SDK modules when openai_prep.agents is
    # first imported; the SDK is only initialised when a factory function is called.
    from agents import ModelSettings  # noqa: PLC0415

    kwargs: dict[str, Any] = {"temperature": model_settings.temperature}
    if model_settings.top_p is not None:
        kwargs["top_p"] = model_settings.top_p
    if model_settings.max_tokens is not None:
        kwargs["max_tokens"] = model_settings.max_tokens
    if model_settings.store is not None:
        kwargs["store"] = model_settings.store
    return ModelSettings(**kwargs)


def _resolve_config(config: AppConfig | None) -> AppConfig:
    return config or default_config()
