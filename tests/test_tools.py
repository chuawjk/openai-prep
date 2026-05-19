import importlib
import sys

from openai_prep.config import SearchConfig


def test_healthhub_search_tool_matches_prototype(monkeypatch):
    calls = []

    class FakeWebSearchTool:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("openai_prep.tools.WebSearchTool", FakeWebSearchTool)

    from openai_prep.tools import create_healthhub_search_tool

    tool = create_healthhub_search_tool()

    assert isinstance(tool, FakeWebSearchTool)
    assert calls == [
        {
            "filters": {"allowed_domains": ["healthhub.sg"]},
            "search_context_size": "medium",
            "user_location": {
                "country": "SG",
                "type": "approximate",
            },
        }
    ]


def test_healthhub_search_tool_uses_configured_search_values(monkeypatch):
    calls = []

    class FakeWebSearchTool:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("openai_prep.tools.WebSearchTool", FakeWebSearchTool)

    from openai_prep.tools import create_healthhub_search_tool

    create_healthhub_search_tool(
        SearchConfig(
            allowed_domains=("healthhub.sg",),
            search_context_size="low",
            user_location={"country": "SG", "type": "approximate"},
        )
    )

    assert calls == [
        {
            "filters": {"allowed_domains": ["healthhub.sg"]},
            "search_context_size": "low",
            "user_location": {
                "country": "SG",
                "type": "approximate",
            },
        }
    ]


def test_config_import_does_not_import_runtime_sdk_modules(monkeypatch):
    for module_name in ("openai_prep.config", "openai", "agents", "guardrails.runtime"):
        sys.modules.pop(module_name, None)

    importlib.import_module("openai_prep.config")

    assert "openai" not in sys.modules
    assert "agents" not in sys.modules
    assert "guardrails.runtime" not in sys.modules
