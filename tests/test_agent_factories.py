from dataclasses import replace

from openai_prep.agents.instructions import (
    INFORMATION_INSTRUCTIONS,
    ORCHESTRATOR_INSTRUCTIONS,
    RECOMMENDER_INSTRUCTIONS,
    REJECTION_INSTRUCTIONS,
    SYNTHESIS_INSTRUCTIONS,
)
from openai_prep.config import ModelSettingsConfig, default_config
from openai_prep.schemas import (
    InformationAgentSchema,
    OrchestratorAgentSchema,
    RecommenderAgentSchema,
)


def test_agent_factories_match_prototype_constructor_arguments(monkeypatch):
    calls = []
    model_settings_calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            model_settings_calls.append(kwargs)

    fake_tool = object()

    # Patch at the agents SDK level so the deferred imports inside _create_agent
    # and _create_model_settings pick up the fakes instead of the real SDK classes.
    import agents as _agents_sdk

    monkeypatch.setattr(_agents_sdk, "Agent", FakeAgent)
    monkeypatch.setattr(_agents_sdk, "ModelSettings", FakeModelSettings)
    monkeypatch.setattr(
        "openai_prep.tools.create_healthhub_search_tool",
        lambda search_config: fake_tool,
    )

    # Deferred imports ensure monkeypatching above takes effect before the
    # module-level SDK references are resolved inside each factory function.
    from openai_prep.agents.information import create_information_agent
    from openai_prep.agents.orchestrator import create_orchestrator_agent
    from openai_prep.agents.recommender import create_recommender_agent
    from openai_prep.agents.rejection import create_rejection_agent
    from openai_prep.agents.synthesis import create_synthesis_agent

    agents = [
        create_orchestrator_agent(),
        create_recommender_agent(),
        create_information_agent(),
        create_synthesis_agent(),
        create_rejection_agent(),
    ]

    assert [agent.kwargs["name"] for agent in agents] == [
        "Orchestrator agent",
        "Recommender agent",
        "Information agent",
        "Synthesis agent",
        "Rejection agent",
    ]
    assert [agent.kwargs["instructions"] for agent in agents] == [
        ORCHESTRATOR_INSTRUCTIONS,
        RECOMMENDER_INSTRUCTIONS,
        INFORMATION_INSTRUCTIONS,
        SYNTHESIS_INSTRUCTIONS,
        REJECTION_INSTRUCTIONS,
    ]
    assert [agent.kwargs["model"] for agent in agents] == ["gpt-4.1"] * 5
    assert agents[0].kwargs["output_type"] is OrchestratorAgentSchema
    assert agents[1].kwargs["output_type"] is RecommenderAgentSchema
    assert agents[2].kwargs["output_type"] is InformationAgentSchema
    assert "output_type" not in agents[3].kwargs
    assert "output_type" not in agents[4].kwargs
    assert agents[2].kwargs["tools"] == [fake_tool]
    assert "tools" not in agents[0].kwargs
    assert "tools" not in agents[1].kwargs
    assert "tools" not in agents[3].kwargs
    assert "tools" not in agents[4].kwargs
    assert model_settings_calls == [
        {"temperature": 0},
        {"temperature": 1, "top_p": 1, "max_tokens": 2048, "store": True},
        {"temperature": 1, "top_p": 1, "max_tokens": 2048, "store": True},
        {"temperature": 1, "top_p": 1, "max_tokens": 2048, "store": True},
        {"temperature": 1, "top_p": 1, "max_tokens": 2048, "store": True},
    ]
    assert len(calls) == 5


def test_information_agent_passes_healthhub_search_config(monkeypatch):
    seen_search_configs = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Patch at the agents SDK level so the deferred imports inside _create_agent
    # and _create_model_settings pick up the fakes instead of the real SDK classes.
    import agents as _agents_sdk

    monkeypatch.setattr(_agents_sdk, "Agent", FakeAgent)
    monkeypatch.setattr(_agents_sdk, "ModelSettings", FakeModelSettings)
    monkeypatch.setattr(
        "openai_prep.tools.create_healthhub_search_tool",
        lambda search_config: seen_search_configs.append(search_config) or "tool",
    )

    # Deferred import ensures monkeypatching above takes effect before the
    # module-level SDK references are resolved inside the factory function.
    from openai_prep.agents.information import create_information_agent

    config = default_config()
    agent = create_information_agent(config)

    assert agent.kwargs["tools"] == ["tool"]
    assert seen_search_configs == [config.search]
    assert config.search.filters() == {"allowed_domains": ["healthhub.sg"]}


def test_agent_factories_use_supplied_config_overrides(monkeypatch):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Patch at the agents SDK level so the deferred imports inside _create_agent
    # and _create_model_settings pick up the fakes instead of the real SDK classes.
    import agents as _agents_sdk

    monkeypatch.setattr(_agents_sdk, "Agent", FakeAgent)
    monkeypatch.setattr(_agents_sdk, "ModelSettings", FakeModelSettings)

    # Deferred import ensures monkeypatching above takes effect before the
    # module-level SDK references are resolved inside the factory function.
    from openai_prep.agents.orchestrator import create_orchestrator_agent

    config = default_config()
    overridden = replace(
        config,
        agents=replace(
            config.agents,
            orchestrator=replace(
                config.agents.orchestrator,
                name="Custom orchestrator",
                model="gpt-test",
                model_settings=ModelSettingsConfig(temperature=0.5, top_p=0.9),
            ),
        ),
    )

    agent = create_orchestrator_agent(overridden)

    assert agent.kwargs["name"] == "Custom orchestrator"
    assert agent.kwargs["model"] == "gpt-test"
    assert agent.kwargs["model_settings"].kwargs == {"temperature": 0.5, "top_p": 0.9}


def test_create_agents_returns_all_agents(monkeypatch):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Patch at the agents SDK level so the deferred imports inside _create_agent
    # and _create_model_settings pick up the fakes instead of the real SDK classes.
    import agents as _agents_sdk

    monkeypatch.setattr(_agents_sdk, "Agent", FakeAgent)
    monkeypatch.setattr(_agents_sdk, "ModelSettings", FakeModelSettings)
    monkeypatch.setattr(
        "openai_prep.tools.create_healthhub_search_tool",
        lambda search_config: object(),
    )

    # Deferred import ensures monkeypatching above takes effect before the
    # module-level SDK references are resolved inside the factory functions.
    from openai_prep.agents.factories import AgentFactories, create_agents

    agents = create_agents()

    assert isinstance(agents, AgentFactories)
    assert agents.orchestrator.name == "Orchestrator agent"
    assert agents.recommender.name == "Recommender agent"
    assert agents.information.name == "Information agent"
    assert agents.synthesis.name == "Synthesis agent"
    assert agents.rejection.name == "Rejection agent"
