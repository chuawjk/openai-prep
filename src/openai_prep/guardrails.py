"""Guardrail logic module.

Provides pure helper functions and thin async wrappers that encapsulate the
prototype guardrail behaviour without importing the guardrail runtime at module
load time.  All ``guardrails.runtime``, ``openai``, and ``agents`` imports are
deferred so that this module can be imported in unit tests without any live
runtime initialisation.

Public API
----------
- ``GuardrailRunnerProtocol`` — callable Protocol for the injectable runner
- ``GuardrailResult`` — TypedDict for the combined guardrail result
- ``FailOutput`` — TypedDict matching ``build_guardrail_fail_output`` return shape
- ``guardrails_has_tripwire(results)`` -> bool
- ``get_guardrail_safe_text(results, fallback_text)`` -> str
- ``build_guardrail_fail_output(results)`` -> FailOutput
- ``scrub_conversation_history(history, config, *, runner)`` (async)
- ``scrub_workflow_input(workflow, input_key, config, *, runner)`` (async)
- ``run_and_apply_guardrails(input_text, config, history, workflow, *, runner, ctx)`` (async)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from typing_extensions import TypedDict

from openai_prep.config import GuardrailsConfig


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------


class FailOutputPII(TypedDict):
    failed: bool
    detected_counts: list[str]


class FailOutputModeration(TypedDict):
    failed: bool
    flagged_categories: list[str]


class FailOutputSimple(TypedDict):
    failed: bool


class FailOutputHallucination(TypedDict):
    failed: bool
    reasoning: str | None
    hallucination_type: str | None
    hallucinated_statements: Any
    verified_statements: Any


class FailOutput(TypedDict):
    pii: FailOutputPII
    moderation: FailOutputModeration
    jailbreak: FailOutputSimple
    hallucination: FailOutputHallucination
    nsfw: FailOutputSimple
    url_filter: FailOutputSimple
    custom_prompt_check: FailOutputSimple
    prompt_injection: FailOutputSimple


class GuardrailResult(TypedDict):
    results: list[Any]
    has_tripwire: bool
    safe_text: str
    fail_output: FailOutput
    pass_output: dict[str, str]


# ---------------------------------------------------------------------------
# Runner Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GuardrailRunnerProtocol(Protocol):
    """Callable protocol for the injectable guardrail runner.

    The real default runner wraps ``guardrails.runtime.run_guardrails``.
    Tests can supply a lightweight mock instead.

    Signature::

        async (ctx, text, mime_type, guardrail_instances, **kwargs) -> list
    """

    async def __call__(
        self,
        ctx: Any,
        text: str,
        mime_type: str,
        guardrail_instances: Any,
        **kwargs: Any,
    ) -> list[Any]: ...


# ---------------------------------------------------------------------------
# Default runner (deferred imports so module-level load stays safe)
# ---------------------------------------------------------------------------


async def _default_runner(
    ctx: Any,
    text: str,
    mime_type: str,
    guardrail_instances: Any,
    **kwargs: Any,
) -> list[Any]:
    """Default runner that delegates to ``guardrails.runtime.run_guardrails``.

    When ``guardrail_instances`` is ``None`` this runner raises; callers that
    use the default runner must supply a pre-built instance bundle via
    ``_load_guardrail_instances``.
    """
    # Deferred import — must not happen at module load time.
    from guardrails.runtime import run_guardrails  # type: ignore[import]

    return await run_guardrails(ctx, text, mime_type, guardrail_instances, **kwargs)


def _make_default_ctx() -> Any:
    """Lazily construct the default guardrail context (deferred OpenAI import)."""
    from openai import AsyncOpenAI  # type: ignore[import]
    from types import SimpleNamespace

    return SimpleNamespace(guardrail_llm=AsyncOpenAI())


def _load_guardrail_instances(config_dict: dict[str, Any]) -> Any:
    """Lazily load and instantiate guardrails from a raw dict bundle.

    Only called on the default (non-injected) runner path.
    """
    from guardrails.runtime import instantiate_guardrails, load_config_bundle  # type: ignore[import]

    return instantiate_guardrails(load_config_bundle(config_dict))


async def _call_runner(
    runner: Any,
    is_default: bool,
    ctx: Any,
    text: str,
    mime_type: str,
    config_dict: dict[str, Any],
    **kwargs: Any,
) -> list[Any]:
    """Dispatch to *runner*, loading guardrail instances only on the default path.

    When an injected (custom) runner is supplied, instances are passed as
    ``None`` — the injected runner is responsible for ignoring or substituting
    that argument (test mocks do this naturally).
    """
    if is_default:
        instances = _load_guardrail_instances(config_dict)
    else:
        instances = None
    return await runner(ctx, text, mime_type, instances, **kwargs)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def guardrails_has_tripwire(results: list[Any] | None) -> bool:
    """Return True if any result has ``tripwire_triggered`` set to True."""
    return any(
        (hasattr(r, "tripwire_triggered") and (r.tripwire_triggered is True))
        for r in (results or [])
    )


def get_guardrail_safe_text(results: list[Any] | None, fallback_text: str) -> str:
    """Return the best scrubbed/checked text from *results*, or *fallback_text*.

    Preference order:
    1. ``checked_text`` from any result's ``info`` dict.
    2. ``anonymized_text`` from the PII guardrail result's ``info`` dict.
    3. *fallback_text*.
    """
    for r in results or []:
        info = (r.info if hasattr(r, "info") else None) or {}
        if isinstance(info, dict) and "checked_text" in info:
            return info.get("checked_text") or fallback_text

    pii_info = next(
        (
            (r.info if hasattr(r, "info") else {})
            for r in (results or [])
            if isinstance((r.info if hasattr(r, "info") else None) or {}, dict)
            and "anonymized_text" in ((r.info if hasattr(r, "info") else None) or {})
        ),
        None,
    )
    if isinstance(pii_info, dict) and "anonymized_text" in pii_info:
        return pii_info.get("anonymized_text") or fallback_text

    return fallback_text


def build_guardrail_fail_output(results: list[Any] | None) -> FailOutput:
    """Build a structured ``FailOutput`` dict from a list of guardrail results.

    Named guardrails that are absent in *results* produce all-False/None/empty
    values so the caller always receives the full key structure.
    """

    def _get(name: str) -> Any:
        for r in results or []:
            info = (r.info if hasattr(r, "info") else None) or {}
            gname = (
                (info.get("guardrail_name") if isinstance(info, dict) else None)
                or (info.get("guardrailName") if isinstance(info, dict) else None)
            )
            if gname == name:
                return r
        return None

    def _tripwire(r: Any) -> bool:
        if r is None:
            return False
        return bool(r.tripwire_triggered)

    def _info(r: Any) -> dict[str, Any]:
        if r is None:
            return {}
        return r.info if hasattr(r, "info") else {}

    pii, mod, jb, hal, nsfw, url, custom, pid = map(
        _get,
        [
            "Contains PII",
            "Moderation",
            "Jailbreak",
            "Hallucination Detection",
            "NSFW Text",
            "URL Filter",
            "Custom Prompt Check",
            "Prompt Injection Detection",
        ],
    )

    jb_info = _info(jb)  # noqa: F841  (kept for symmetry / future use)
    hal_info = _info(hal)
    nsfw_info = _info(nsfw)  # noqa: F841
    url_info = _info(url)  # noqa: F841
    custom_info = _info(custom)  # noqa: F841
    pid_info = _info(pid)  # noqa: F841
    mod_info = _info(mod)
    pii_info = _info(pii)

    detected_entities = (
        pii_info.get("detected_entities") if isinstance(pii_info, dict) else {}
    )
    pii_counts: list[str] = []
    if isinstance(detected_entities, dict):
        for k, v in detected_entities.items():
            if isinstance(v, list):
                pii_counts.append(f"{k}:{len(v)}")

    flagged_categories: list[str] = (
        (mod_info.get("flagged_categories") if isinstance(mod_info, dict) else None)
        or []
    )

    return {
        "pii": {
            "failed": (len(pii_counts) > 0) or _tripwire(pii),
            "detected_counts": pii_counts,
        },
        "moderation": {
            "failed": _tripwire(mod) or (len(flagged_categories) > 0),
            "flagged_categories": flagged_categories,
        },
        "jailbreak": {"failed": _tripwire(jb)},
        "hallucination": {
            "failed": _tripwire(hal),
            "reasoning": (
                hal_info.get("reasoning") if isinstance(hal_info, dict) else None
            ),
            "hallucination_type": (
                hal_info.get("hallucination_type")
                if isinstance(hal_info, dict)
                else None
            ),
            "hallucinated_statements": (
                hal_info.get("hallucinated_statements")
                if isinstance(hal_info, dict)
                else None
            ),
            "verified_statements": (
                hal_info.get("verified_statements")
                if isinstance(hal_info, dict)
                else None
            ),
        },
        "nsfw": {"failed": _tripwire(nsfw)},
        "url_filter": {"failed": _tripwire(url)},
        "custom_prompt_check": {"failed": _tripwire(custom)},
        "prompt_injection": {"failed": _tripwire(pid)},
    }


# ---------------------------------------------------------------------------
# Async scrubbing helpers
# ---------------------------------------------------------------------------


async def scrub_conversation_history(
    history: list[Any] | None,
    config: GuardrailsConfig,
    *,
    runner: Any = None,
) -> None:
    """Scrub PII from every ``input_text`` part in *history* in-place.

    Uses the *runner* callable to execute the PII-only guardrail.  If *runner*
    is ``None`` the default runtime runner is used (lazy import).
    """
    try:
        pii = next(
            (g for g in config.guardrails if g.name == "Contains PII"),
            None,
        )
        if not pii:
            return

        pii_only_dict = {"guardrails": [pii.as_dict()]}
        is_default = runner is None
        _runner = _default_runner if is_default else runner
        _ctx = _make_default_ctx() if is_default else None

        for msg in history or []:
            content = (msg or {}).get("content") or []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "input_text"
                    and isinstance(part.get("text"), str)
                ):
                    res = await _call_runner(
                        _runner,
                        is_default,
                        _ctx,
                        part["text"],
                        "text/plain",
                        pii_only_dict,
                        suppress_tripwire=True,
                        raise_guardrail_errors=True,
                    )
                    part["text"] = get_guardrail_safe_text(res, part["text"])
    except Exception:
        pass


async def scrub_workflow_input(
    workflow: dict[str, Any] | None,
    input_key: str,
    config: GuardrailsConfig,
    *,
    runner: Any = None,
) -> None:
    """Scrub PII from ``workflow[input_key]`` in-place.

    Uses the *runner* callable to execute the PII-only guardrail.  If *runner*
    is ``None`` the default runtime runner is used (lazy import).
    """
    try:
        pii = next(
            (g for g in config.guardrails if g.name == "Contains PII"),
            None,
        )
        if not pii:
            return
        if not isinstance(workflow, dict):
            return
        value = workflow.get(input_key)
        if not isinstance(value, str):
            return

        pii_only_dict = {"guardrails": [pii.as_dict()]}
        is_default = runner is None
        _runner = _default_runner if is_default else runner
        _ctx = _make_default_ctx() if is_default else None

        res = await _call_runner(
            _runner,
            is_default,
            _ctx,
            value,
            "text/plain",
            pii_only_dict,
            suppress_tripwire=True,
            raise_guardrail_errors=True,
        )
        workflow[input_key] = get_guardrail_safe_text(res, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Top-level combined guardrail entry point
# ---------------------------------------------------------------------------


async def run_and_apply_guardrails(
    input_text: str,
    config: GuardrailsConfig,
    history: list[Any] | None,
    workflow: dict[str, Any] | None,
    *,
    runner: Any = None,
    ctx: Any = None,
) -> GuardrailResult:
    """Run all guardrails against *input_text*, scrub PII if needed, and return
    a structured ``GuardrailResult``.

    Parameters
    ----------
    input_text:
        Raw user input to screen.
    config:
        ``GuardrailsConfig`` instance from ``config.py``.
    history:
        Mutable conversation history list; scrubbed in-place when PII masking
        is active.
    workflow:
        Mutable workflow dict; ``input_as_text`` / ``input_text`` keys scrubbed
        in-place when PII masking is active.
    runner:
        Injectable async callable matching ``GuardrailRunnerProtocol``.  When
        ``None`` the default runtime runner is used (lazy import).
    ctx:
        Guardrail context object (must carry a ``guardrail_llm`` attribute).
        When ``None`` a fresh ``AsyncOpenAI()`` context is constructed lazily.
    """
    is_default = runner is None
    _runner = _default_runner if is_default else runner
    _ctx = ctx if ctx is not None else (_make_default_ctx() if is_default else None)

    full_config_dict = config.as_dict()
    results = await _call_runner(
        _runner,
        is_default,
        _ctx,
        input_text,
        "text/plain",
        full_config_dict,
        suppress_tripwire=True,
        raise_guardrail_errors=True,
    )

    # Check whether the PII guardrail is configured in non-blocking mode.
    mask_pii = (
        next(
            (
                g
                for g in config.guardrails
                if g.name == "Contains PII"
                and g.config.get("block") is False
            ),
            None,
        )
        is not None
    )

    if mask_pii:
        await scrub_conversation_history(history, config, runner=runner)
        await scrub_workflow_input(workflow, "input_as_text", config, runner=runner)
        await scrub_workflow_input(workflow, "input_text", config, runner=runner)

    has_tripwire = guardrails_has_tripwire(results)
    safe_text = get_guardrail_safe_text(results, input_text)
    fail_output = build_guardrail_fail_output(results or [])
    pass_output: dict[str, str] = {
        "safe_text": (get_guardrail_safe_text(results, input_text) or input_text)
    }

    return {
        "results": results,
        "has_tripwire": has_tripwire,
        "safe_text": safe_text,
        "fail_output": fail_output,
        "pass_output": pass_output,
    }
