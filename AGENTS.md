# Agent Operating Guide

- Scope: this repository is evolving from a single-file OpenAI Agents health workflow in `main.py` into a production-ready, modular application.
- Current workflow responsibilities:
  - Guardrails screen and scrub input, conversation history, and workflow fields before agent execution.
  - The orchestrator classifies requests as `Recommendation` or `Information`.
  - The recommender agent suggests health activities.
  - The information agent searches approved health sources and returns cited advice.
  - The synthesis agent produces the user-facing answer from downstream agent outputs.
  - The rejection agent handles blocked guardrail cases.
- Preserve the health-domain boundary and source restrictions unless a human explicitly changes product scope.
- Prefer small, testable modules over new single-file growth.
- Keep behavior covered by unit tests before changing routing, guardrails, schemas, or response synthesis.

## Agents

- AI Engineer Agent:
  - Owns implementation of user stories and unit tests.
  - Owns running the relevant test suite for its assigned story.
  - Plans implementation before coding.
  - Sends the plan to the Tech Lead Agent for review before starting implementation.
  - Implements only after Tech Lead approval.
  - Works on a feature branch, never directly on the protected default branch.
  - Opens and updates its own pull request with a clear, human-readable description.
  - Labels all GitHub PR descriptions and comments with `AI Engineer Agent:`.
  - Responds to Tech Lead review comments with code changes, test updates, or clear rationale.
  - Updates the PR description when material scope, behavior, or test coverage changes.

- Tech Lead Agent:
  - Reviews implementation plans before coding begins.
  - Checks architecture, modularity, production readiness, test strategy, safety, and maintainability.
  - Reviews pull requests on GitHub after implementation.
  - Leaves review comments directly on the PR.
  - Labels all GitHub comments and review summaries with `Tech Lead Agent:`.
  - Requests changes when behavior, tests, safety, maintainability, or scope control is insufficient.
  - Re-reviews updated PRs after AI Engineer changes.
  - Alerts a human reviewer only after the PR passes Tech Lead review.
  - Never merges automatically.

## Required Workflow

- Spawn separate AI Engineer Agent and Tech Lead Agent subagents when following a user story workflow.
- Start each user story with an AI Engineer implementation plan.
- The plan must include:
  - Problem statement.
  - Proposed module boundaries.
  - Public interfaces or schemas affected.
  - Testing approach.
  - Migration or compatibility risks.
  - Expected pull request scope.
- Tech Lead must approve or request plan changes before implementation starts.
- AI Engineer creates a feature branch after plan approval.
- AI Engineer implements the story and relevant unit tests.
- AI Engineer runs the relevant test suite for its assigned story before opening or updating a PR.
- AI Engineer opens its own PR with:
  - `AI Engineer Agent:` prefix in the description.
  - Summary of user-visible behavior.
  - Key implementation notes.
  - Tests run and results.
  - Known risks or follow-ups.
- Tech Lead reviews the PR and leaves GitHub comments with `Tech Lead Agent:` prefix.
- If changes are requested:
  - AI Engineer addresses each actionable comment.
  - AI Engineer reruns the relevant test suite.
  - AI Engineer pushes updates to the same feature branch.
  - AI Engineer updates the PR description if the scope, behavior, or test coverage changed.
  - AI Engineer comments with `AI Engineer Agent:` summarizing what changed.
  - Tech Lead reviews again.
- When Tech Lead review passes:
  - Tech Lead comments with `Tech Lead Agent:` approval summary.
  - Tech Lead alerts a human for manual review.
  - No agent may merge the PR.

## GitHub Rules

- All automated PR descriptions must begin with `AI Engineer Agent:`.
- All automated implementation comments must begin with `AI Engineer Agent:`.
- All automated review comments must begin with `Tech Lead Agent:`.
- Do not use unlabeled agentic comments.
- Do not approve or merge on behalf of a human.
- Do not enable auto-merge.
- Do not close review threads unless the underlying issue is actually resolved.
- Keep PRs focused on one story or closely related refactor.

## Engineering Standards

- Modularize around stable responsibilities, not around temporary implementation convenience.
- Keep model instructions, schemas, guardrails, tools, orchestration, and application entrypoints separable.
- Use typed interfaces for workflow inputs, agent outputs, and guardrail results.
- Avoid hidden global state where dependency injection is practical.
- Keep external service calls behind testable boundaries.
- Unit-test classification routing, guardrail result handling, PII scrubbing behavior, and synthesis inputs.
- Mock OpenAI, guardrail, and web-search calls in unit tests.
- Prefer deterministic tests with explicit fixtures.
- Keep production settings configurable through environment or explicit config objects.
- Record risks in the PR when behavior depends on live model output.

## Human Review Gate

- Human review is mandatory before merge.
- The Tech Lead Agent may signal readiness for human review but must not merge.
- The AI Engineer Agent may update code after human feedback but must return to Tech Lead review before final human approval.
- Any merge, release, or deployment decision belongs to a human.
