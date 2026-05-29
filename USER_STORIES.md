# User Story Roadmap

This roadmap turns the current Agent Builder prototype into a production-ready, modular health-agent application while practicing the AI Engineer / Tech Lead delivery workflow.

Target package layout:

```text
src/openai_prep/
  __init__.py
  config.py
  schemas.py
  tools.py
  guardrails.py
  workflow.py
  agents/
    __init__.py
    orchestrator.py
    recommender.py
    information.py
    synthesis.py
    rejection.py
```

`agent_builder.py` remains the generated prototype/reference until the modular workflow reaches parity. `main.py` should eventually become a thin compatibility wrapper or example entry point.

## 1. Establish Project Baseline

- As a maintainer, I want project metadata, dependencies, and test tooling declared so agents can run consistent local checks.
- Acceptance:
  - Required dependencies are declared.
  - pytest is configured.
  - A minimal smoke test runs without live API calls.
  - README documents setup and test commands.
- Tests:
  - Basic smoke test using mocks or no runtime imports.

## 2. Add Local Environment Template And Runtime Setup Docs

- As a developer, I want local environment setup documented before runtime code is moved so every later story has clear verification assumptions.
- Acceptance:
  - Add `.env.example` with `OPENAI_API_KEY=` and any safe optional local settings.
  - Ensure `.env` is ignored.
  - Document `uv sync --all-extras --dev` and `uv run pytest`.
  - Document that CI uses mocked tests only and does not require secrets.
  - Document optional live verification requirements without adding live tests to CI.
- Tests:
  - No automated tests required; verify docs and ignore rules.

## 3. Create Package Skeleton

- As an AI Engineer, I want the target package structure in place before moving behavior so module destinations are explicit.
- Acceptance:
  - Create `src/openai_prep/` and `src/openai_prep/agents/`.
  - Add package `__init__.py` files.
  - Configure packaging so `openai_prep` imports under uv/pytest.
  - Do not move runtime behavior yet.
- Tests:
  - Import smoke test for `openai_prep`.

## 4. Extract Typed Schemas

- As an AI Engineer, I want workflow inputs and agent output schemas in `src/openai_prep/schemas.py` so behavior contracts are explicit.
- Acceptance:
  - Move Pydantic models out of `agent_builder.py` into `schemas.py`.
  - Preserve existing schema names or provide compatibility aliases where needed.
  - Do not change runtime behavior.
- Tests:
  - Valid and invalid `WorkflowInput` validation.
  - Orchestrator, recommender, and information output schema validation.

## 5. Add Configuration Layer

- As an operator, I want models, trace metadata, search domain, search location, and guardrail defaults centralized in `src/openai_prep/config.py`.
- Acceptance:
  - Defaults match the current prototype behavior.
  - Model names, HealthHub domain, Singapore search location, trace metadata, and guardrail settings are represented in typed config.
  - Config can be passed explicitly by later factories/services.
- Tests:
  - Default config snapshot.
  - Override behavior for model name, search domain, and trace metadata.

## 6. Modularize Agent Construction

- As an AI Engineer, I want each OpenAI Agent factory in `src/openai_prep/agents/` so agent definitions are independent from orchestration.
- Acceptance:
  - Implement factories for orchestrator, recommender, information, synthesis, and rejection agents.
  - Put web-search tool construction in `src/openai_prep/tools.py`.
  - Preserve existing agent names, instructions, models, output schemas, and HealthHub restriction.
  - Factories accept config rather than relying on scattered constants.
- Tests:
  - Factory tests for expected names, schemas, models, and tool restrictions.
  - Mock or inspect agent objects without making live API calls.

## 7. Modularize Guardrail Handling

- As a maintainer, I want guardrail config, execution, PII scrubbing, and failure-output construction isolated in `src/openai_prep/guardrails.py`.
- Acceptance:
  - Guardrail logic is callable without running the full workflow.
  - Existing PII masking and tripwire behavior are preserved.
  - External guardrail execution is injectable or mockable.
- Tests:
  - Tripwire detection.
  - Safe-text selection.
  - PII anonymized-text preference.
  - Fail-output construction.
  - Mocked guardrail runner paths.

## 8. Introduce Workflow Orchestrator Service

- As a product owner, I want the health workflow orchestrated by `src/openai_prep/workflow.py` so routing and execution flow are clear and unit-testable.
- Acceptance:
  - `run_workflow` delegates to a workflow service.
  - Guardrail rejection, recommendation route, and information route remain behaviorally equivalent.
  - Runner execution and guardrail execution are mockable.
- Tests:
  - Mocked rejection path.
  - Mocked recommendation path.
  - Mocked information path.
  - Synthesis invocation checks.

## 9. Create Application Entry Point

- As an integrator, I want a clean public entry point for invoking the workflow so future CLI/API surfaces do not depend on internals.
- Acceptance:
  - Public API exposes workflow input and async execution from `openai_prep`.
  - `main.py` becomes a thin compatibility wrapper or example entry point.
  - `agent_builder.py` remains available as prototype reference unless explicitly retired.
- Tests:
  - Public entry point calls orchestrator with expected config.
  - Public entry point returns expected output shape.

## 10. Add Optional Live Verification Command

- As a developer, I want an explicit local-only way to verify the live agent workflow without making CI depend on secrets or network calls.
- Acceptance:
  - Add a documented command or small script for one live workflow invocation.
  - Require `OPENAI_API_KEY` from local environment.
  - Skip or fail clearly when required env vars are missing.
  - Keep live verification out of default pytest and CI.
- Tests:
  - Unit test the missing-env behavior if a script is added.
  - Do not add live API tests to CI.

## 11. End-To-End Mocked Workflow Coverage

- As a maintainer, I want mocked end-to-end tests over the modular system so refactors do not break routing behavior.
- Acceptance:
  - Mocked E2E tests cover health recommendation, health information with sources, non-health rejection, and PII scrubbing.
  - Tests do not make live network or OpenAI calls.
  - Reuse mocks introduced with each module story.
- Tests:
  - Full mocked workflow scenarios.

## 12. Production Readiness Review

- As a Tech Lead, I want a final review pass over architecture, tests, configuration, docs, and local/live verification before human release review.
- Acceptance:
  - Tech Lead produces a review checklist.
  - AI Engineer fixes any findings in follow-up PRs.
  - Final state is marked ready for human review, not automatically merged.
- Tests:
  - Full test suite run documented in the PR.
  - Optional live verification result documented when credentials are available.

## 13. End-To-End Evals With OpenAI Evals API

- As a product owner, I want automated quality evals over the live workflow using the OpenAI Evals API so that routing correctness and response quality are continuously measurable.
- Acceptance:
  - A JSONL test dataset at `evals/data/health_queries.jsonl` covers all three routing paths: information, recommendation, and rejection.
  - An eval definition script at `evals/create_eval.py` registers an eval object via `client.evals.create`, with:
    - A `data_source_config` schema that includes `input_text` (the query) and `expected_category` (`"Recommendation"`, `"Information"`, or `"Rejection"`).
    - A `string_check` grader that compares `{{ sample.output_text }}` against expected routing behavior.
  - An eval run script at `evals/run_eval.py` submits a run via `client.evals.runs.create` referencing the registered eval and a JSONL file upload.
  - A README section documents how to create, run, and inspect evals locally (`uv run python evals/create_eval.py`, `uv run python evals/run_eval.py`).
  - Eval scripts require `OPENAI_API_KEY` from the environment and fail clearly when absent.
  - Eval scripts are excluded from the default `uv run pytest` run and CI.
- Tests:
  - Unit tests for dataset loading and eval script argument parsing (no live API calls).
  - Do not add live eval runs to CI.

## Delivery Rules

- Implement each story as a small feature branch and focused PR.
- Start each story with an AI Engineer implementation plan.
- Get Tech Lead approval before implementation.
- Add story-specific mocks and fixtures as part of the story that needs them.
- Label GitHub PR descriptions and comments with the agentic author.
- Never merge automatically; human review is mandatory.
