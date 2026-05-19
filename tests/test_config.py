import dataclasses
import importlib
import os
import sys
from dataclasses import replace
from types import MappingProxyType

import pytest

from openai_prep.config import (
    AgentConfig,
    AppConfig,
    GuardrailConfig,
    GuardrailsConfig,
    SearchConfig,
    TraceConfig,
    default_config,
)


def test_default_config_matches_prototype_snapshot():
    config = default_config()

    assert config.agents.orchestrator.name == "Orchestrator agent"
    assert config.agents.orchestrator.model == "gpt-4.1"
    assert config.agents.orchestrator.model_settings.temperature == 0
    assert config.agents.orchestrator.model_settings.top_p is None
    assert config.agents.orchestrator.model_settings.max_tokens is None
    assert config.agents.orchestrator.model_settings.store is None

    for agent in (
        config.agents.recommender,
        config.agents.information,
        config.agents.synthesis,
        config.agents.rejection,
    ):
        assert agent.model == "gpt-4.1"
        assert agent.model_settings.temperature == 1
        assert agent.model_settings.top_p == 1
        assert agent.model_settings.max_tokens == 2048
        assert agent.model_settings.store is True

    assert config.search.allowed_domains == ("healthhub.sg",)
    assert config.search.search_context_size == "medium"
    assert config.search.user_location == {
        "country": "SG",
        "type": "approximate",
    }
    assert config.search.filters() == {"allowed_domains": ["healthhub.sg"]}

    assert config.trace.trace_source == "agent-builder"
    assert config.trace.workflow_id == "wf_6a0a98b97c108190a2fb1e0b1642a4fe0de15210713bebbe"
    assert config.trace.metadata() == {
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_6a0a98b97c108190a2fb1e0b1642a4fe0de15210713bebbe",
    }

    guardrail_payload = config.guardrails.as_dict()
    assert guardrail_payload == {
        "guardrails": [
            {
                "name": "Contains PII",
                "config": {
                    "block": False,
                    "detect_encoded_pii": True,
                    "entities": [
                        "CREDIT_CARD",
                        "US_BANK_NUMBER",
                        "US_PASSPORT",
                        "US_SSN",
                    ],
                },
            },
            {
                "name": "NSFW Text",
                "config": {
                    "model": "gpt-4.1-mini",
                    "confidence_threshold": 0.7,
                },
            },
            {
                "name": "Moderation",
                "config": {
                    "categories": [
                        "sexual/minors",
                        "hate/threatening",
                        "harassment/threatening",
                        "self-harm/instructions",
                        "violence/graphic",
                        "illicit/violent",
                    ],
                },
            },
            {
                "name": "Jailbreak",
                "config": {
                    "model": "gpt-4.1-mini",
                    "confidence_threshold": 0.7,
                },
            },
            {
                "name": "Prompt Injection Detection",
                "config": {
                    "model": "gpt-4.1-mini",
                    "confidence_threshold": 0.7,
                },
            },
            {
                "name": "Custom Prompt Check",
                "config": {
                    "system_prompt_details": (
                        "You are a health coach chatbot. Raise the guardrail if "
                        "the user request is not related to health matters"
                    ),
                    "model": "gpt-4.1-mini",
                    "confidence_threshold": 0.7,
                },
            },
        ]
    }


def test_config_supports_declarative_overrides():
    config = default_config()

    overridden_model = replace(config.agents.orchestrator, model="gpt-4.1-mini")
    overridden_search = replace(config.search, allowed_domains=("example.org",))
    overridden_trace = TraceConfig(
        trace_source="unit-test",
        workflow_id="wf_test",
    )
    overridden = replace(
        config,
        agents=replace(config.agents, orchestrator=overridden_model),
        search=overridden_search,
        trace=overridden_trace,
    )

    assert overridden.agents.orchestrator.model == "gpt-4.1-mini"
    assert overridden.search.allowed_domains == ("example.org",)
    assert overridden.search.filters() == {"allowed_domains": ["example.org"]}
    assert overridden.trace.metadata() == {
        "__trace_source__": "unit-test",
        "workflow_id": "wf_test",
    }

    assert config.agents.orchestrator.model == "gpt-4.1"
    assert config.search.allowed_domains == ("healthhub.sg",)
    assert config.trace.trace_source == "agent-builder"


def test_config_is_immutable_and_does_not_share_mutable_state():
    first = default_config()
    second = default_config()

    assert dataclasses.is_dataclass(first)
    assert isinstance(first, AppConfig)
    assert isinstance(first.agents.orchestrator, AgentConfig)
    assert isinstance(first.search, SearchConfig)
    assert isinstance(first.trace, TraceConfig)
    assert isinstance(first.guardrails, GuardrailsConfig)
    assert isinstance(first.guardrails.guardrails[0], GuardrailConfig)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.search.search_context_size = "low"

    with pytest.raises(TypeError):
        first.search.user_location["country"] = "US"

    with pytest.raises(TypeError):
        first.guardrails.guardrails[0].config["block"] = True

    assert isinstance(first.search.allowed_domains, tuple)
    assert isinstance(first.search.user_location, MappingProxyType)
    assert isinstance(first.guardrails.guardrails, tuple)
    assert first.guardrails.guardrails is not second.guardrails.guardrails
    assert first.guardrails.guardrails[0].config is not second.guardrails.guardrails[0].config

    first_payload = first.guardrails.as_dict()
    first_payload["guardrails"][0]["config"]["entities"].append("EMAIL")
    assert "EMAIL" not in first.guardrails.guardrails[0].config["entities"]
    assert "EMAIL" not in second.guardrails.guardrails[0].config["entities"]


def test_trace_metadata_returns_fresh_dicts():
    trace = default_config().trace

    first = trace.metadata()
    second = trace.metadata()

    assert first == second
    assert first is not second

    first["workflow_id"] = "changed"
    assert second["workflow_id"] == "wf_6a0a98b97c108190a2fb1e0b1642a4fe0de15210713bebbe"
    assert trace.workflow_id == "wf_6a0a98b97c108190a2fb1e0b1642a4fe0de15210713bebbe"


def test_importing_config_is_safe_without_credentials_or_runtime_initialization(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for module_name in ("openai_prep.config", "openai", "agents", "guardrails.runtime"):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("openai_prep.config")

    assert module.default_config().search.allowed_domains == ("healthhub.sg",)
    assert "openai" not in sys.modules
    assert "agents" not in sys.modules
    assert "guardrails.runtime" not in sys.modules
    assert os.environ.get("OPENAI_API_KEY") is None
