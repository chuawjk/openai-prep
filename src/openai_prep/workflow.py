"""Workflow orchestration module for OpenAI Prep.

Provides ``run_workflow`` — a module-level async function that replicates the
routing logic from ``agent_builder.py`` using the modular config, schemas,
guardrails, and agent factories from this package.

All ``agents`` SDK imports (``Runner``, ``RunConfig``, ``trace``) are deferred
inside function bodies so that importing this module does NOT load the SDK,
openai, or guardrails.runtime.
"""

from __future__ import annotations

from typing import Any

from openai_prep.agents import create_agents
from openai_prep.config import AppConfig, default_config
from openai_prep.guardrails import run_and_apply_guardrails
from openai_prep.schemas import WorkflowInput


async def run_workflow(
    workflow_input: WorkflowInput,
    *,
    config: AppConfig | None = None,
    runner_fn: Any = None,
    guardrail_runner: Any = None,
) -> dict[str, Any]:
    """Run the full health-coach workflow and return ``{"output_text": ...}``.

    Parameters
    ----------
    workflow_input:
        Typed workflow input (carries ``input_as_text``).
    config:
        ``AppConfig`` to use.  ``default_config()`` is used when ``None``.
    runner_fn:
        Injectable async callable used instead of ``Runner.run`` for every
        agent invocation.  Signature matches ``Runner.run``::

            async runner_fn(agent, *, input, run_config, ...) -> RunResult

        When ``None`` the real ``Runner.run`` is used (deferred import).
    guardrail_runner:
        Injectable async runner forwarded to ``run_and_apply_guardrails``.
        When ``None`` the default guardrails runtime runner is used.
    """
    # Deferred import: trace must not load at module import time.
    from agents import trace  # noqa: PLC0415

    resolved_config = config or default_config()

    with trace("New agent"):
        workflow = workflow_input.model_dump()
        conversation_history: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": workflow["input_as_text"],
                    }
                ],
            }
        ]

        guardrails_result = await run_and_apply_guardrails(
            workflow["input_as_text"],
            resolved_config.guardrails,
            conversation_history,
            workflow,
            runner=guardrail_runner,
        )

        _runner = runner_fn if runner_fn is not None else _default_runner_fn
        run_config = _make_run_config(resolved_config)
        agents = create_agents(resolved_config)

        if guardrails_result["has_tripwire"]:
            # --- Rejection path ---
            result = await _runner(
                agents.rejection,
                input=[*conversation_history],
                run_config=run_config,
            )
            conversation_history.extend(
                [item.to_input_item() for item in result.new_items]
            )
            return {"output_text": result.final_output_as(str)}

        else:
            # --- Pass path: classify first ---
            orchestrator_input: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": workflow["input_as_text"],
                        }
                    ],
                }
            ]
            orchestrator_result = await _runner(
                agents.orchestrator,
                input=orchestrator_input,
                run_config=run_config,
            )
            category: str = orchestrator_result.final_output.category

            if category == "Recommendation":
                # --- Recommendation path ---
                recommender_result = await _runner(
                    agents.recommender,
                    input=[*conversation_history],
                    run_config=run_config,
                )
                conversation_history.extend(
                    [item.to_input_item() for item in recommender_result.new_items]
                )

                synthesis_result = await _runner(
                    agents.synthesis,
                    input=[*conversation_history],
                    run_config=run_config,
                )
                conversation_history.extend(
                    [item.to_input_item() for item in synthesis_result.new_items]
                )
                return {"output_text": synthesis_result.final_output_as(str)}

            else:
                # --- Information path ---
                information_result = await _runner(
                    agents.information,
                    input=[*conversation_history],
                    run_config=run_config,
                )
                conversation_history.extend(
                    [item.to_input_item() for item in information_result.new_items]
                )

                synthesis_result = await _runner(
                    agents.synthesis,
                    input=[*conversation_history],
                    run_config=run_config,
                )
                conversation_history.extend(
                    [item.to_input_item() for item in synthesis_result.new_items]
                )
                return {"output_text": synthesis_result.final_output_as(str)}


def _make_run_config(config: AppConfig) -> Any:
    """Build a ``RunConfig`` from *config*.

    The ``agents`` SDK import is deferred so this function (and its caller)
    can be exercised without loading the SDK.
    """
    from agents import RunConfig  # noqa: PLC0415

    return RunConfig(trace_metadata=config.trace.metadata())


async def _default_runner_fn(agent: Any, *, input: Any, run_config: Any, **kwargs: Any) -> Any:  # noqa: A002
    """Default runner that delegates to ``Runner.run``.

    Deferred import prevents SDK load at module import time.
    """
    from agents import Runner  # noqa: PLC0415

    return await Runner.run(agent, input=input, run_config=run_config, **kwargs)
