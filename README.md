# OpenAI Prep

Health-domain OpenAI Agents workflow prototype.

## Local Setup

Install dependencies with uv:

```bash
uv sync
```

Create a local environment file when you need to run against live OpenAI services:

```bash
cp .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`. Do not commit `.env`; it is ignored by git.

## Tests

Run the test suite with:

```bash
uv run pytest
```

Tests should use mocks or deterministic fixtures for OpenAI, guardrail, and web-search behavior. The CI test suite is expected to run mocked tests only and must not require `OPENAI_API_KEY` or any other secret.

## Live Verification

To run one live end-to-end workflow invocation:

```bash
export OPENAI_API_KEY=<your-key>
uv run python -m openai_prep.verify
```

This requires a valid `OPENAI_API_KEY` in your shell environment and makes real calls to the OpenAI API. Do not add this command to CI or automated test runs.
