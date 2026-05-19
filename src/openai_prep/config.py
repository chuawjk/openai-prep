"""Declarative application configuration defaults.

This module intentionally contains data only. It must stay free of OpenAI,
Agents SDK, guardrail runtime, and other live runtime object imports so later
factories can consume the values without causing initialization at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


AGENT_MODEL = "gpt-4.1"
GUARDRAIL_MODEL = "gpt-4.1-mini"
TRACE_SOURCE = "agent-builder"
WORKFLOW_ID = "wf_6a0a98b97c108190a2fb1e0b1642a4fe0de15210713bebbe"
HEALTH_DOMAIN_SYSTEM_PROMPT = (
    "You are a health coach chatbot. Raise the guardrail if the user request "
    "is not related to health matters"
)


@dataclass(frozen=True, slots=True)
class ModelSettingsConfig:
    """Declarative model settings matching Agents SDK ModelSettings fields."""

    temperature: float
    top_p: float | None = None
    max_tokens: int | None = None
    store: bool | None = None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Declarative agent configuration consumed by future agent factories."""

    name: str
    model: str
    model_settings: ModelSettingsConfig


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Declarative web search configuration."""

    allowed_domains: tuple[str, ...] = ("healthhub.sg",)
    search_context_size: str = "medium"
    user_location: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(
            {
                "country": "SG",
                "type": "approximate",
            }
        )
    )

    def filters(self) -> dict[str, list[str]]:
        """Return fresh mutable filter data for runtime constructors."""

        return {"allowed_domains": list(self.allowed_domains)}

    def user_location_dict(self) -> dict[str, str]:
        """Return a fresh user location dict for runtime constructors."""

        return dict(self.user_location)


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Declarative trace metadata defaults."""

    trace_source: str = TRACE_SOURCE
    workflow_id: str = WORKFLOW_ID

    def metadata(self) -> dict[str, str]:
        """Return fresh trace metadata for each run configuration."""

        return {
            "__trace_source__": self.trace_source,
            "workflow_id": self.workflow_id,
        }


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Declarative guardrail entry matching the prototype config shape."""

    name: str
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return fresh mutable data for guardrail runtime loading."""

        return {"name": self.name, "config": _to_mutable(self.config)}


@dataclass(frozen=True, slots=True)
class GuardrailsConfig:
    """Collection of declarative guardrail entries."""

    guardrails: tuple[GuardrailConfig, ...]

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return fresh mutable data for guardrail runtime loading."""

        return {"guardrails": [guardrail.as_dict() for guardrail in self.guardrails]}


@dataclass(frozen=True, slots=True)
class AgentsConfig:
    """Declarative configuration for all prototype agents."""

    orchestrator: AgentConfig
    recommender: AgentConfig
    information: AgentConfig
    synthesis: AgentConfig
    rejection: AgentConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    agents: AgentsConfig
    search: SearchConfig
    trace: TraceConfig
    guardrails: GuardrailsConfig


def default_config() -> AppConfig:
    """Build the prototype-equivalent declarative default configuration."""

    return AppConfig(
        agents=AgentsConfig(
            orchestrator=AgentConfig(
                name="Orchestrator agent",
                model=AGENT_MODEL,
                model_settings=ModelSettingsConfig(temperature=0),
            ),
            recommender=AgentConfig(
                name="Recommender agent",
                model=AGENT_MODEL,
                model_settings=_generative_model_settings(),
            ),
            information=AgentConfig(
                name="Information agent",
                model=AGENT_MODEL,
                model_settings=_generative_model_settings(),
            ),
            synthesis=AgentConfig(
                name="Synthesis agent",
                model=AGENT_MODEL,
                model_settings=_generative_model_settings(),
            ),
            rejection=AgentConfig(
                name="Rejection agent",
                model=AGENT_MODEL,
                model_settings=_generative_model_settings(),
            ),
        ),
        search=SearchConfig(),
        trace=TraceConfig(),
        guardrails=GuardrailsConfig(
            guardrails=(
                GuardrailConfig(
                    name="Contains PII",
                    config=MappingProxyType(
                        {
                            "block": False,
                            "detect_encoded_pii": True,
                            "entities": (
                                "CREDIT_CARD",
                                "US_BANK_NUMBER",
                                "US_PASSPORT",
                                "US_SSN",
                            ),
                        }
                    ),
                ),
                GuardrailConfig(
                    name="NSFW Text",
                    config=MappingProxyType(
                        {
                            "model": GUARDRAIL_MODEL,
                            "confidence_threshold": 0.7,
                        }
                    ),
                ),
                GuardrailConfig(
                    name="Moderation",
                    config=MappingProxyType(
                        {
                            "categories": (
                                "sexual/minors",
                                "hate/threatening",
                                "harassment/threatening",
                                "self-harm/instructions",
                                "violence/graphic",
                                "illicit/violent",
                            ),
                        }
                    ),
                ),
                GuardrailConfig(
                    name="Jailbreak",
                    config=MappingProxyType(
                        {
                            "model": GUARDRAIL_MODEL,
                            "confidence_threshold": 0.7,
                        }
                    ),
                ),
                GuardrailConfig(
                    name="Prompt Injection Detection",
                    config=MappingProxyType(
                        {
                            "model": GUARDRAIL_MODEL,
                            "confidence_threshold": 0.7,
                        }
                    ),
                ),
                GuardrailConfig(
                    name="Custom Prompt Check",
                    config=MappingProxyType(
                        {
                            "system_prompt_details": HEALTH_DOMAIN_SYSTEM_PROMPT,
                            "model": GUARDRAIL_MODEL,
                            "confidence_threshold": 0.7,
                        }
                    ),
                ),
            )
        ),
    )


def _generative_model_settings() -> ModelSettingsConfig:
    return ModelSettingsConfig(
        temperature=1,
        top_p=1,
        max_tokens=2048,
        store=True,
    )


def _to_mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_mutable(item) for item in value]
    return value
