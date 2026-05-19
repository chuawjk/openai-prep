"""Tests for openai_prep.workflow.

All agents SDK, openai, and guardrails.runtime imports are mocked so the tests
run without any live service credentials or SDK initialisation.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from openai_prep.config import default_config
from openai_prep.schemas import WorkflowInput


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_item(text: str = "agent output") -> MagicMock:
    """Return a fake RunResultStreamEvent-like item with .to_input_item()."""
    item = MagicMock()
    item.to_input_item.return_value = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    return item


@pytest.fixture
def fake_run_result():
    """Factory that builds a configurable fake RunResult."""

    def _make(final_output_str: str = "result text", items: list[Any] | None = None):
        result = MagicMock()
        result.final_output = MagicMock()
        result.final_output_as = MagicMock(return_value=final_output_str)
        result.new_items = items if items is not None else [_make_item(final_output_str)]
        return result

    return _make


@pytest.fixture
def no_tripwire_guardrail():
    """AsyncMock for run_and_apply_guardrails that always passes."""
    return AsyncMock(
        return_value={
            "results": [],
            "has_tripwire": False,
            "safe_text": "safe text",
            "fail_output": {},
            "pass_output": {"safe_text": "safe text"},
        }
    )


@pytest.fixture
def tripwire_guardrail():
    """AsyncMock for run_and_apply_guardrails that always trips."""
    return AsyncMock(
        return_value={
            "results": [],
            "has_tripwire": True,
            "safe_text": "blocked",
            "fail_output": {"jailbreak": {"failed": True}},
            "pass_output": {},
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Rejection path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rejection_path_calls_only_rejection_agent(
    fake_run_result, tripwire_guardrail
):
    """When guardrails trip, only rejection_agent is called; orchestrator never runs."""
    rejection_output = "I cannot help with that."
    rejection_result = fake_run_result(final_output_str=rejection_output)
    runner_fn = AsyncMock(return_value=rejection_result)

    workflow_input = WorkflowInput(input_as_text="something harmful")

    with patch(
        "openai_prep.workflow.run_and_apply_guardrails",
        tripwire_guardrail,
    ), patch(
        "openai_prep.workflow.create_agents",
    ) as mock_create_agents, patch(
        "openai_prep.workflow._make_run_config",
    ):
        agents_mock = MagicMock()
        mock_create_agents.return_value = agents_mock

        from openai_prep.workflow import run_workflow

        result = await run_workflow(
            workflow_input,
            runner_fn=runner_fn,
            guardrail_runner=MagicMock(),
        )

    # runner_fn called exactly once — with rejection agent
    assert runner_fn.call_count == 1
    call_kwargs = runner_fn.call_args
    assert call_kwargs.args[0] is agents_mock.rejection

    # orchestrator was never invoked
    orchestrator_calls = [
        c for c in runner_fn.call_args_list if c.args[0] is agents_mock.orchestrator
    ]
    assert len(orchestrator_calls) == 0

    # return value has output_text
    assert "output_text" in result
    assert result["output_text"] == rejection_output


# ---------------------------------------------------------------------------
# Test 2: Recommendation path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_recommendation_path_calls_orchestrator_recommender_synthesis(
    fake_run_result, no_tripwire_guardrail
):
    """Recommendation path: runner_fn called 3× in correct order; synthesis is last."""
    # Orchestrator result with category="Recommendation"
    orchestrator_result = fake_run_result(items=[])
    orchestrator_result.final_output.category = "Recommendation"

    recommender_item = _make_item("recommender output")
    recommender_result = fake_run_result(items=[recommender_item])

    synthesis_item = _make_item("synthesis output")
    synthesis_result = fake_run_result(
        final_output_str="final synthesis text", items=[synthesis_item]
    )

    call_order: list[str] = []

    agents_mock = MagicMock()

    async def fake_runner(agent, *, input, run_config, **kwargs):
        if agent is agents_mock.orchestrator:
            call_order.append("orchestrator")
            return orchestrator_result
        elif agent is agents_mock.recommender:
            call_order.append("recommender")
            return recommender_result
        elif agent is agents_mock.synthesis:
            call_order.append("synthesis")
            return synthesis_result
        raise ValueError(f"Unexpected agent: {agent}")

    workflow_input = WorkflowInput(input_as_text="recommend me an activity")

    with patch(
        "openai_prep.workflow.run_and_apply_guardrails",
        no_tripwire_guardrail,
    ), patch(
        "openai_prep.workflow.create_agents",
        return_value=agents_mock,
    ), patch(
        "openai_prep.workflow._make_run_config",
    ):
        from openai_prep.workflow import run_workflow

        result = await run_workflow(
            workflow_input,
            runner_fn=fake_runner,
            guardrail_runner=MagicMock(),
        )

    # Exactly 3 calls in order
    assert call_order == ["orchestrator", "recommender", "synthesis"]

    # Orchestrator received a FRESH list (not the shared conversation_history)
    # — verify it has exactly one user message with the original input text
    orchestrator_call = None
    # We need to capture calls; use a wrapper approach instead
    # The call_order list confirms ordering; verify synthesis was last
    assert call_order[-1] == "synthesis"

    # Return value contains output_text
    assert "output_text" in result
    assert result["output_text"] == "final synthesis text"


@pytest.mark.anyio
async def test_recommendation_path_history_threading(
    fake_run_result, no_tripwire_guardrail
):
    """Synthesis receives conversation_history extended by recommender's new_items."""
    orchestrator_result = fake_run_result(items=[])
    orchestrator_result.final_output.category = "Recommendation"

    recommender_item = _make_item("rec output")
    recommender_result = fake_run_result(items=[recommender_item])
    synthesis_result = fake_run_result(final_output_str="synthesised", items=[])

    agents_mock = MagicMock()
    synthesis_input_captured: list[Any] = []

    async def fake_runner(agent, *, input, run_config, **kwargs):
        if agent is agents_mock.orchestrator:
            return orchestrator_result
        elif agent is agents_mock.recommender:
            return recommender_result
        elif agent is agents_mock.synthesis:
            synthesis_input_captured.extend(input)
            return synthesis_result
        raise ValueError(f"Unexpected agent: {agent}")

    workflow_input = WorkflowInput(input_as_text="suggest an activity")

    with patch(
        "openai_prep.workflow.run_and_apply_guardrails",
        no_tripwire_guardrail,
    ), patch(
        "openai_prep.workflow.create_agents",
        return_value=agents_mock,
    ), patch(
        "openai_prep.workflow._make_run_config",
    ):
        from openai_prep.workflow import run_workflow

        await run_workflow(
            workflow_input,
            runner_fn=fake_runner,
            guardrail_runner=MagicMock(),
        )

    # Synthesis input must include the recommender's to_input_item() output
    recommender_item_dict = recommender_item.to_input_item()
    assert recommender_item_dict in synthesis_input_captured


@pytest.mark.anyio
async def test_recommendation_path_orchestrator_receives_fresh_list(
    fake_run_result, no_tripwire_guardrail
):
    """Orchestrator must receive a fresh list, not conversation_history."""
    orchestrator_result = fake_run_result(items=[])
    orchestrator_result.final_output.category = "Recommendation"

    recommender_item = _make_item("rec output")
    recommender_result = fake_run_result(items=[recommender_item])
    synthesis_result = fake_run_result(final_output_str="done", items=[])

    agents_mock = MagicMock()
    orchestrator_input_captured: list[Any] = []

    async def fake_runner(agent, *, input, run_config, **kwargs):
        if agent is agents_mock.orchestrator:
            orchestrator_input_captured.extend(input)
            return orchestrator_result
        elif agent is agents_mock.recommender:
            return recommender_result
        elif agent is agents_mock.synthesis:
            return synthesis_result
        raise ValueError(f"Unexpected agent: {agent}")

    workflow_input = WorkflowInput(input_as_text="suggest something")

    with patch(
        "openai_prep.workflow.run_and_apply_guardrails",
        no_tripwire_guardrail,
    ), patch(
        "openai_prep.workflow.create_agents",
        return_value=agents_mock,
    ), patch(
        "openai_prep.workflow._make_run_config",
    ):
        from openai_prep.workflow import run_workflow

        await run_workflow(
            workflow_input,
            runner_fn=fake_runner,
            guardrail_runner=MagicMock(),
        )

    # Orchestrator input is a fresh list with a single user message
    assert len(orchestrator_input_captured) == 1
    assert orchestrator_input_captured[0]["role"] == "user"
    assert orchestrator_input_captured[0]["content"][0]["text"] == "suggest something"


# ---------------------------------------------------------------------------
# Test 3: Information path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_information_path_calls_orchestrator_information_synthesis(
    fake_run_result, no_tripwire_guardrail
):
    """Information path: orchestrator→information→synthesis; recommender never called."""
    orchestrator_result = fake_run_result(items=[])
    orchestrator_result.final_output.category = "Information"

    information_item = _make_item("info output")
    information_result = fake_run_result(items=[information_item])
    synthesis_result = fake_run_result(final_output_str="info synthesis", items=[])

    call_order: list[str] = []
    agents_mock = MagicMock()

    async def fake_runner(agent, *, input, run_config, **kwargs):
        if agent is agents_mock.orchestrator:
            call_order.append("orchestrator")
            return orchestrator_result
        elif agent is agents_mock.information:
            call_order.append("information")
            return information_result
        elif agent is agents_mock.synthesis:
            call_order.append("synthesis")
            return synthesis_result
        elif agent is agents_mock.recommender:
            call_order.append("recommender")
            raise AssertionError("recommender must not be called on Information path")
        raise ValueError(f"Unexpected agent: {agent}")

    workflow_input = WorkflowInput(input_as_text="what are the benefits of walking?")

    with patch(
        "openai_prep.workflow.run_and_apply_guardrails",
        no_tripwire_guardrail,
    ), patch(
        "openai_prep.workflow.create_agents",
        return_value=agents_mock,
    ), patch(
        "openai_prep.workflow._make_run_config",
    ):
        from openai_prep.workflow import run_workflow

        result = await run_workflow(
            workflow_input,
            runner_fn=fake_runner,
            guardrail_runner=MagicMock(),
        )

    assert call_order == ["orchestrator", "information", "synthesis"]
    assert "recommender" not in call_order

    assert "output_text" in result
    assert result["output_text"] == "info synthesis"


@pytest.mark.anyio
async def test_information_path_history_threading(
    fake_run_result, no_tripwire_guardrail
):
    """Synthesis receives conversation_history extended by information's new_items."""
    orchestrator_result = fake_run_result(items=[])
    orchestrator_result.final_output.category = "Information"

    information_item = _make_item("info output")
    information_result = fake_run_result(items=[information_item])
    synthesis_result = fake_run_result(final_output_str="synthesised info", items=[])

    agents_mock = MagicMock()
    synthesis_input_captured: list[Any] = []

    async def fake_runner(agent, *, input, run_config, **kwargs):
        if agent is agents_mock.orchestrator:
            return orchestrator_result
        elif agent is agents_mock.information:
            return information_result
        elif agent is agents_mock.synthesis:
            synthesis_input_captured.extend(input)
            return synthesis_result
        raise ValueError(f"Unexpected agent: {agent}")

    workflow_input = WorkflowInput(input_as_text="tell me about hydration")

    with patch(
        "openai_prep.workflow.run_and_apply_guardrails",
        no_tripwire_guardrail,
    ), patch(
        "openai_prep.workflow.create_agents",
        return_value=agents_mock,
    ), patch(
        "openai_prep.workflow._make_run_config",
    ):
        from openai_prep.workflow import run_workflow

        await run_workflow(
            workflow_input,
            runner_fn=fake_runner,
            guardrail_runner=MagicMock(),
        )

    # Synthesis input must include information agent's to_input_item() output
    information_item_dict = information_item.to_input_item()
    assert information_item_dict in synthesis_input_captured


# ---------------------------------------------------------------------------
# Test 4: _make_run_config unit test
# ---------------------------------------------------------------------------


def test_make_run_config_returns_run_config_with_trace_metadata():
    """_make_run_config must return RunConfig(trace_metadata=config.trace.metadata())."""
    config = default_config()

    # We test via a captured mock so we don't need live SDK
    captured_kwargs: dict[str, Any] = {}

    class FakeRunConfig:
        def __init__(self, **kwargs: Any):
            captured_kwargs.update(kwargs)

    with patch("openai_prep.workflow._make_run_config") as mock_make:
        # Call the real implementation by importing directly
        pass

    # Now test with real deferred import, replacing RunConfig
    with patch.dict(sys.modules, {"agents": MagicMock()}):
        # Force reload so the deferred import picks up the mock
        import agents as agents_mod  # noqa: PLC0415

        agents_mod.RunConfig = FakeRunConfig

        from openai_prep.workflow import _make_run_config  # noqa: PLC0415

        _make_run_config(config)

    expected_metadata = config.trace.metadata()
    assert captured_kwargs.get("trace_metadata") == expected_metadata


# ---------------------------------------------------------------------------
# Test 5: Import safety
# ---------------------------------------------------------------------------


def test_importing_workflow_does_not_load_agents_openai_or_guardrails_runtime(
    monkeypatch,
):
    """Importing openai_prep.workflow must NOT load agents, openai, or
    guardrails.runtime as side effects."""
    # Remove the module from sys.modules so re-importing is fresh.
    for mod in (
        "openai_prep.workflow",
        "agents",
        "openai",
        "guardrails.runtime",
    ):
        sys.modules.pop(mod, None)

    importlib.import_module("openai_prep.workflow")

    assert "agents" not in sys.modules, "agents SDK was eagerly imported"
    assert "openai" not in sys.modules, "openai was eagerly imported"
    assert "guardrails.runtime" not in sys.modules, (
        "guardrails.runtime was eagerly imported"
    )
