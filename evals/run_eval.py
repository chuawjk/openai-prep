"""Run an OpenAI eval using pre-computed routing labels.

Usage:
    uv run python evals/run_eval.py --eval-id <id> [--dataset PATH]

Requires OPENAI_API_KEY to be set in the environment.

For each item in the dataset:
  1. Run guardrails; if tripwire fires → category = "Rejection".
  2. Otherwise call the orchestrator agent directly to get the category.
Pre-computed results are uploaded as a JSONL file and submitted to the eval run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_DEFAULT_DATASET = Path(__file__).parent / "data" / "health_queries.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OpenAI eval using pre-computed routing labels."
    )
    parser.add_argument(
        "--eval-id",
        required=True,
        help="ID of the eval to run (created by create_eval.py)",
    )
    parser.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help=f"Path to a JSONL dataset file (default: {_DEFAULT_DATASET})",
    )
    return parser.parse_args(argv)


async def _compute_routing(input_text: str) -> str:
    """Compute routing category for a single input text.

    Calls guardrails first; if tripwire fires returns "Rejection".
    Otherwise calls the orchestrator agent and returns its category.

    All SDK imports are deferred so this module is importable without the SDK.
    """
    from agents import RunConfig, Runner  # noqa: PLC0415
    from openai_prep.agents import create_agents  # noqa: PLC0415
    from openai_prep.config import default_config  # noqa: PLC0415
    from openai_prep.guardrails import run_and_apply_guardrails  # noqa: PLC0415

    config = default_config()

    guardrails_result = await run_and_apply_guardrails(
        input_text,
        config.guardrails,
        history=None,
        workflow=None,
    )

    if guardrails_result["has_tripwire"]:
        return "Rejection"

    agents = create_agents(config)
    run_config = RunConfig(trace_metadata=config.trace.metadata())

    orchestrator_result = await Runner.run(
        agents.orchestrator,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": input_text,
                    }
                ],
            }
        ],
        run_config=run_config,
    )

    category: str = orchestrator_result.final_output.category
    return category


async def run(argv: list[str] | None = None) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your shell before running evals:\n"
            "  export OPENAI_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args(argv)
    eval_id: str = args.eval_id
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"Error: dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Load dataset items
    items: list[dict] = []
    with dataset_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Computing routing for {len(items)} items...", file=sys.stderr)

    # Pre-compute routing labels
    results: list[dict] = []
    for item in items:
        input_text: str = item["input_text"]
        output_category = await _compute_routing(input_text)
        results.append(
            {
                "item": {
                    "input_text": input_text,
                    "expected_category": item["expected_category"],
                },
                "sample": {
                    "output_text": output_category,
                },
            }
        )

    # Write results to a temporary JSONL file for upload
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name
        for row in results:
            tmp_file.write(json.dumps(row) + "\n")

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=api_key)

    print("Uploading results file...", file=sys.stderr)
    with open(tmp_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="evals")

    os.unlink(tmp_path)

    print(f"Submitting eval run for eval_id={eval_id}...", file=sys.stderr)
    run_obj = client.evals.runs.create(
        eval_id,
        name="Health query routing run",
        data_source={
            "type": "jsonl",
            "source": {"type": "file_id", "id": uploaded_file.id},
        },
    )

    print(run_obj.id)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(argv))


if __name__ == "__main__":
    main()
