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
