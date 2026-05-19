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

## Evals

End-to-end evaluation of the health workflow output quality using the OpenAI Evals API.

### Dataset

The evaluation dataset is at `evals/data/health_queries.jsonl`. It contains at least 9 queries (minimum 3 per query type: information, recommendation, and rejection queries). Each row has only `input_text` — no pre-labelled categories.

### Create an eval definition

```bash
export OPENAI_API_KEY=<your-key>
uv run python evals/create_eval.py [--eval-name "Health Query Routing"]
```

Prints the created eval ID to stdout. Store this ID for the run step.

The eval uses two `label_model` graders powered by `gpt-4o-mini`:

- **Relevance** — judges whether the response meaningfully addresses the query, including polite refusals for out-of-scope or harmful requests.
- **Politeness** — judges whether the response is polite and appropriate in tone.

### Run an eval

```bash
export OPENAI_API_KEY=<your-key>
uv run python evals/run_eval.py --eval-id <eval-id> [--dataset evals/data/health_queries.jsonl]
```

For each item in the dataset, the script calls `run_workflow` to get the full assistant response (~3–4 live OpenAI calls per row). The collected outputs are uploaded and submitted as a pre-computed eval run. Prints the run ID to stdout.

Both scripts require `OPENAI_API_KEY` and make real calls to the OpenAI API. Do not add these commands to CI or automated test runs.
