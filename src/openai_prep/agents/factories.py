"""Agent factories for the OpenAI Prep workflow."""

from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, ModelSettings

from openai_prep.agents.instructions import (
    INFORMATION_INSTRUCTIONS,
    ORCHESTRATOR_INSTRUCTIONS,
    RECOMMENDER_INSTRUCTIONS,
    REJECTION_INSTRUCTIONS,
    SYNTHESIS_INSTRUCTIONS,
)
from openai_prep.config import AgentConfig, AppConfig, ModelSettingsConfig, default_config
from openai_prep.schemas import (
    InformationAgentSchema,
    OrchestratorAgentSchema,
    RecommenderAgentSchema,
)
from openai_prep.tools import create_healthhub_search_tool


@dataclass(frozen=True, slots=True)
class AgentFactories:
    """Container for all workflow agents."""

    orchestrator: Agent
    recommender: Agent
    information: Agent
    synthesis: Agent
    rejection: Agent


def create_orchestrator_agent(config: AppConfig | None = None) -> Agent:
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.orchestrator,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        output_type=OrchestratorAgentSchema,
    )


def create_recommender_agent(config: AppConfig | None = None) -> Agent:
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.recommender,
        instructions=RECOMMENDER_INSTRUCTIONS,
        output_type=RecommenderAgentSchema,
    )


def create_information_agent(config: AppConfig | None = None) -> Agent:
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.information,
        instructions=INFORMATION_INSTRUCTIONS,
        tools=[create_healthhub_search_tool(resolved_config.search)],
        output_type=InformationAgentSchema,
    )


def create_synthesis_agent(config: AppConfig | None = None) -> Agent:
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.synthesis,
        instructions=SYNTHESIS_INSTRUCTIONS,
    )


def create_rejection_agent(config: AppConfig | None = None) -> Agent:
    resolved_config = _resolve_config(config)
    return _create_agent(
        resolved_config.agents.rejection,
        instructions=REJECTION_INSTRUCTIONS,
    )


def create_agents(config: AppConfig | None = None) -> AgentFactories:
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
    tools: list | None = None,
) -> Agent:
    kwargs = {
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
    kwargs = {"temperature": model_settings.temperature}
    if model_settings.top_p is not None:
        kwargs["top_p"] = model_settings.top_p
    if model_settings.max_tokens is not None:
        kwargs["max_tokens"] = model_settings.max_tokens
    if model_settings.store is not None:
        kwargs["store"] = model_settings.store
    return ModelSettings(**kwargs)


def _resolve_config(config: AppConfig | None) -> AppConfig:
    return config or default_config()
