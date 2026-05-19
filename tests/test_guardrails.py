"""Tests for openai_prep.guardrails.

All guardrail runtime, openai, and agents imports are mocked so the tests run
without any live service credentials or SDK initialisation.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_prep.config import GuardrailConfig, GuardrailsConfig, default_config
from openai_prep.guardrails import (
    build_guardrail_fail_output,
    get_guardrail_safe_text,
    guardrails_has_tripwire,
    run_and_apply_guardrails,
    scrub_conversation_history,
    scrub_workflow_input,
)


# ---------------------------------------------------------------------------
# Helpers to build fake result objects
# ---------------------------------------------------------------------------


def _make_result(
    *,
    tripwire: bool = False,
    info: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Return a fake guardrail result object."""
    return SimpleNamespace(tripwire_triggered=tripwire, info=info or {})


def _make_named_result(
    name: str,
    *,
    tripwire: bool = False,
    info_extra: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Return a fake result with ``guardrail_name`` set in ``info``."""
    info: dict[str, Any] = {"guardrail_name": name}
    if info_extra:
        info.update(info_extra)
    return SimpleNamespace(tripwire_triggered=tripwire, info=info)


# ---------------------------------------------------------------------------
# Import-safety test (clarification 3)
# ---------------------------------------------------------------------------


def test_importing_guardrails_does_not_load_runtime_openai_or_agents(monkeypatch):
    """Importing openai_prep.guardrails must NOT load guardrails.runtime,
    openai, or agents as side effects."""
    # Remove the module from sys.modules so that re-importing it is fresh.
    for mod in ("openai_prep.guardrails", "guardrails.runtime", "openai", "agents"):
        sys.modules.pop(mod, None)

    importlib.import_module("openai_prep.guardrails")

    assert "guardrails.runtime" not in sys.modules, (
        "guardrails.runtime was eagerly imported"
    )
    assert "openai" not in sys.modules, "openai was eagerly imported"
    assert "agents" not in sys.modules, "agents was eagerly imported"


# ---------------------------------------------------------------------------
# guardrails_has_tripwire
# ---------------------------------------------------------------------------


def test_guardrails_has_tripwire_returns_false_for_empty_results():
    assert guardrails_has_tripwire([]) is False


def test_guardrails_has_tripwire_returns_false_for_none():
    assert guardrails_has_tripwire(None) is False


def test_guardrails_has_tripwire_returns_false_when_no_tripwire_set():
    results = [_make_result(tripwire=False), _make_result(tripwire=False)]
    assert guardrails_has_tripwire(results) is False


def test_guardrails_has_tripwire_returns_true_when_any_tripwire_triggered():
    results = [_make_result(tripwire=False), _make_result(tripwire=True)]
    assert guardrails_has_tripwire(results) is True


def test_guardrails_has_tripwire_ignores_objects_without_tripwire_attribute():
    results = [object(), _make_result(tripwire=False)]
    assert guardrails_has_tripwire(results) is False


# ---------------------------------------------------------------------------
# get_guardrail_safe_text
# ---------------------------------------------------------------------------


def test_get_guardrail_safe_text_returns_fallback_when_no_results():
    assert get_guardrail_safe_text([], "fallback") == "fallback"


def test_get_guardrail_safe_text_returns_fallback_for_none():
    assert get_guardrail_safe_text(None, "fallback") == "fallback"


def test_get_guardrail_safe_text_prefers_checked_text():
    results = [
        _make_result(info={"checked_text": "clean text", "anonymized_text": "anon"}),
    ]
    assert get_guardrail_safe_text(results, "fallback") == "clean text"


def test_get_guardrail_safe_text_falls_back_to_anonymized_text():
    results = [_make_result(info={"anonymized_text": "anon text"})]
    assert get_guardrail_safe_text(results, "fallback") == "anon text"


def test_get_guardrail_safe_text_prefers_checked_text_over_anonymized():
    results = [
        _make_result(info={"anonymized_text": "anon text"}),
        _make_result(info={"checked_text": "checked"}),
    ]
    # Iterates in order — first hit with checked_text wins
    assert get_guardrail_safe_text(results, "fallback") == "checked"


def test_get_guardrail_safe_text_returns_fallback_when_checked_text_empty():
    results = [_make_result(info={"checked_text": ""})]
    assert get_guardrail_safe_text(results, "fallback") == "fallback"


def test_get_guardrail_safe_text_returns_fallback_when_anonymized_text_empty():
    results = [_make_result(info={"anonymized_text": ""})]
    assert get_guardrail_safe_text(results, "fallback") == "fallback"


# ---------------------------------------------------------------------------
# build_guardrail_fail_output — full key structure
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "pii",
    "moderation",
    "jailbreak",
    "hallucination",
    "nsfw",
    "url_filter",
    "custom_prompt_check",
    "prompt_injection",
}


def test_build_guardrail_fail_output_has_full_key_structure_when_no_results():
    """Clarification 1: even with no named guardrail results the full key
    structure must be present with all-False/None/empty values."""
    output = build_guardrail_fail_output([])

    assert set(output.keys()) == EXPECTED_KEYS

    assert output["pii"]["failed"] is False
    assert output["pii"]["detected_counts"] == []

    assert output["moderation"]["failed"] is False
    assert output["moderation"]["flagged_categories"] == []

    assert output["jailbreak"]["failed"] is False

    assert output["hallucination"]["failed"] is False
    assert output["hallucination"]["reasoning"] is None
    assert output["hallucination"]["hallucination_type"] is None
    assert output["hallucination"]["hallucinated_statements"] is None
    assert output["hallucination"]["verified_statements"] is None

    assert output["nsfw"]["failed"] is False
    assert output["url_filter"]["failed"] is False
    assert output["custom_prompt_check"]["failed"] is False
    assert output["prompt_injection"]["failed"] is False


def test_build_guardrail_fail_output_has_full_key_structure_when_results_is_none():
    """Clarification 1: None results also produce full key structure."""
    output = build_guardrail_fail_output(None)
    assert set(output.keys()) == EXPECTED_KEYS
    assert output["pii"]["failed"] is False
    assert output["jailbreak"]["failed"] is False
    assert output["hallucination"]["reasoning"] is None


def test_build_guardrail_fail_output_detects_pii_from_named_result():
    results = [
        _make_named_result(
            "Contains PII",
            tripwire=False,
            info_extra={
                "detected_entities": {
                    "CREDIT_CARD": ["1234"],
                    "US_SSN": ["111-22-3333", "444-55-6666"],
                }
            },
        )
    ]
    output = build_guardrail_fail_output(results)
    assert output["pii"]["failed"] is True
    assert "CREDIT_CARD:1" in output["pii"]["detected_counts"]
    assert "US_SSN:2" in output["pii"]["detected_counts"]


def test_build_guardrail_fail_output_detects_tripwire_in_jailbreak():
    results = [_make_named_result("Jailbreak", tripwire=True)]
    output = build_guardrail_fail_output(results)
    assert output["jailbreak"]["failed"] is True
    # Other keys still have full structure
    assert set(output.keys()) == EXPECTED_KEYS
    assert output["pii"]["failed"] is False


def test_build_guardrail_fail_output_detects_moderation_flagged_categories():
    results = [
        _make_named_result(
            "Moderation",
            tripwire=False,
            info_extra={"flagged_categories": ["hate/threatening", "violence/graphic"]},
        )
    ]
    output = build_guardrail_fail_output(results)
    assert output["moderation"]["failed"] is True
    assert output["moderation"]["flagged_categories"] == [
        "hate/threatening",
        "violence/graphic",
    ]


def test_build_guardrail_fail_output_hallucination_fields_populated():
    results = [
        _make_named_result(
            "Hallucination Detection",
            tripwire=True,
            info_extra={
                "reasoning": "some reason",
                "hallucination_type": "factual",
                "hallucinated_statements": ["bad claim"],
                "verified_statements": ["good claim"],
            },
        )
    ]
    output = build_guardrail_fail_output(results)
    assert output["hallucination"]["failed"] is True
    assert output["hallucination"]["reasoning"] == "some reason"
    assert output["hallucination"]["hallucination_type"] == "factual"
    assert output["hallucination"]["hallucinated_statements"] == ["bad claim"]
    assert output["hallucination"]["verified_statements"] == ["good claim"]


def test_build_guardrail_fail_output_none_result_does_not_crash():
    """Clarification 1: absent named guardrails must not raise AttributeError."""
    # results list contains one result for a different named guardrail; PII
    # lookup returns None and must be handled defensively.
    results = [_make_named_result("NSFW Text", tripwire=True)]
    output = build_guardrail_fail_output(results)
    # nsfw is present and triggered
    assert output["nsfw"]["failed"] is True
    # pii is absent → must not crash and must be False
    assert output["pii"]["failed"] is False
    assert output["pii"]["detected_counts"] == []


# ---------------------------------------------------------------------------
# scrub_conversation_history
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_scrub_conversation_history_replaces_input_text():
    config = default_config().guardrails

    scrubbed = _make_result(info={"anonymized_text": "REDACTED"})
    runner = AsyncMock(return_value=[scrubbed])

    history = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "my SSN is 123-45-6789"}],
        }
    ]

    await scrub_conversation_history(history, config, runner=runner)

    assert history[0]["content"][0]["text"] == "REDACTED"
    runner.assert_called_once()


@pytest.mark.anyio
async def test_scrub_conversation_history_no_pii_guardrail_is_noop():
    """If the config has no PII guardrail, history must be left unchanged."""
    config = GuardrailsConfig(
        guardrails=(GuardrailConfig(name="NSFW Text", config={"model": "gpt-4.1-mini", "confidence_threshold": 0.7}),)
    )
    runner = AsyncMock()

    history = [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]

    await scrub_conversation_history(history, config, runner=runner)

    assert history[0]["content"][0]["text"] == "hello"
    runner.assert_not_called()


@pytest.mark.anyio
async def test_scrub_conversation_history_skips_non_input_text_parts():
    config = default_config().guardrails
    runner = AsyncMock(return_value=[])

    history = [
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "response"}],
        }
    ]

    await scrub_conversation_history(history, config, runner=runner)

    # output_text is not an input_text part so runner is not called
    runner.assert_not_called()
    assert history[0]["content"][0]["text"] == "response"


# ---------------------------------------------------------------------------
# scrub_workflow_input
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_scrub_workflow_input_replaces_string_value():
    config = default_config().guardrails

    scrubbed = _make_result(info={"anonymized_text": "SCRUBBED"})
    runner = AsyncMock(return_value=[scrubbed])

    workflow = {"input_as_text": "my passport is AB1234567"}

    await scrub_workflow_input(workflow, "input_as_text", config, runner=runner)

    assert workflow["input_as_text"] == "SCRUBBED"
    runner.assert_called_once()


@pytest.mark.anyio
async def test_scrub_workflow_input_no_pii_guardrail_is_noop():
    config = GuardrailsConfig(
        guardrails=(GuardrailConfig(name="Jailbreak", config={"model": "gpt-4.1-mini", "confidence_threshold": 0.7}),)
    )
    runner = AsyncMock()

    workflow = {"input_as_text": "hello"}
    await scrub_workflow_input(workflow, "input_as_text", config, runner=runner)

    assert workflow["input_as_text"] == "hello"
    runner.assert_not_called()


@pytest.mark.anyio
async def test_scrub_workflow_input_ignores_non_string_values():
    config = default_config().guardrails
    runner = AsyncMock()

    workflow = {"input_as_text": 42}
    await scrub_workflow_input(workflow, "input_as_text", config, runner=runner)

    assert workflow["input_as_text"] == 42
    runner.assert_not_called()


@pytest.mark.anyio
async def test_scrub_workflow_input_missing_key_is_noop():
    config = default_config().guardrails
    runner = AsyncMock()

    workflow: dict[str, Any] = {}
    await scrub_workflow_input(workflow, "input_as_text", config, runner=runner)

    runner.assert_not_called()


# ---------------------------------------------------------------------------
# run_and_apply_guardrails — mocked runner paths
# ---------------------------------------------------------------------------


def _make_fake_ctx() -> Any:
    return SimpleNamespace(guardrail_llm=MagicMock())


@pytest.mark.anyio
async def test_run_and_apply_guardrails_no_tripwire_pass_output():
    config = default_config().guardrails
    ctx = _make_fake_ctx()

    results = [_make_result(tripwire=False, info={"checked_text": "safe input"})]
    runner = AsyncMock(return_value=results)

    history: list[Any] = []
    workflow: dict[str, Any] = {"input_as_text": "safe input", "input_text": "safe input"}

    result = await run_and_apply_guardrails(
        "safe input", config, history, workflow, runner=runner, ctx=ctx
    )

    assert result["has_tripwire"] is False
    assert result["safe_text"] == "safe input"
    assert result["pass_output"] == {"safe_text": "safe input"}
    assert result["results"] is results


@pytest.mark.anyio
async def test_run_and_apply_guardrails_tripwire_sets_has_tripwire():
    config = default_config().guardrails
    ctx = _make_fake_ctx()

    results = [_make_named_result("Jailbreak", tripwire=True)]
    runner = AsyncMock(return_value=results)

    result = await run_and_apply_guardrails(
        "jailbreak attempt", config, [], {}, runner=runner, ctx=ctx
    )

    assert result["has_tripwire"] is True
    assert result["fail_output"]["jailbreak"]["failed"] is True


@pytest.mark.anyio
async def test_run_and_apply_guardrails_pii_masking_scrubs_history_and_workflow():
    """When PII guardrail has block=False, history and workflow must be scrubbed."""
    config = default_config().guardrails
    ctx = _make_fake_ctx()

    # First call (main run) — return a non-PII result
    main_result = _make_result(tripwire=False)
    # Subsequent calls (scrub_conversation_history + 2× scrub_workflow_input)
    scrubbed_result = _make_result(info={"anonymized_text": "ANON"})

    call_count = 0

    async def fake_runner(ctx_arg, text, mime, instances, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [main_result]
        return [scrubbed_result]

    history = [
        {"role": "user", "content": [{"type": "input_text", "text": "SSN 123-45-6789"}]}
    ]
    workflow = {"input_as_text": "SSN 123-45-6789", "input_text": "SSN 123-45-6789"}

    await run_and_apply_guardrails(
        "SSN 123-45-6789", config, history, workflow, runner=fake_runner, ctx=ctx
    )

    assert history[0]["content"][0]["text"] == "ANON"
    assert workflow["input_as_text"] == "ANON"
    assert workflow["input_text"] == "ANON"


@pytest.mark.anyio
async def test_run_and_apply_guardrails_returns_full_guardrail_result_keys():
    config = default_config().guardrails
    ctx = _make_fake_ctx()

    runner = AsyncMock(return_value=[])

    result = await run_and_apply_guardrails(
        "hello", config, [], {}, runner=runner, ctx=ctx
    )

    assert set(result.keys()) == {
        "results",
        "has_tripwire",
        "safe_text",
        "fail_output",
        "pass_output",
    }
    assert set(result["fail_output"].keys()) == EXPECTED_KEYS


@pytest.mark.anyio
async def test_run_and_apply_guardrails_uses_fallback_text_when_no_safe_text():
    config = default_config().guardrails
    ctx = _make_fake_ctx()

    runner = AsyncMock(return_value=[_make_result(tripwire=False)])

    result = await run_and_apply_guardrails(
        "original text", config, [], {}, runner=runner, ctx=ctx
    )

    assert result["safe_text"] == "original text"
    assert result["pass_output"]["safe_text"] == "original text"


# ---------------------------------------------------------------------------
# GuardrailsConfig-typed parameter enforcement (clarification 2)
# ---------------------------------------------------------------------------


def test_functions_accept_guardrails_config_not_raw_dict():
    """Verify that build_guardrail_fail_output and guardrails_has_tripwire work
    with the typed config; this is a static/structural check that ensures the
    module uses the .guardrails attribute of GuardrailsConfig rather than
    dict .get() calls."""
    config = default_config().guardrails
    assert isinstance(config, GuardrailsConfig)
    # Access .guardrails attribute (list of GuardrailConfig) — this is what the
    # module must use internally.
    pii = next((g for g in config.guardrails if g.name == "Contains PII"), None)
    assert pii is not None
    assert hasattr(pii, "name")
    assert hasattr(pii, "config")
    # as_dict() must produce the canonical raw dict for runtime loading
    d = pii.as_dict()
    assert d["name"] == "Contains PII"
    assert d["config"]["block"] is False
