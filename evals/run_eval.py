"""Run an OpenAI eval by collecting full workflow outputs.

Usage:
    uv run python evals/run_eval.py --eval-id <id> [--dataset PATH]

Requires OPENAI_API_KEY to be set in the environment.

For each item in the dataset, calls run_workflow to get the full assistant
response (~3-4 live OpenAI calls per row). Pre-computed results are uploaded
as a JSONL file and submitted to the eval run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile

_DEFAULT_DATASET = "evals/data/health_queries.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OpenAI eval using full workflow outputs."
    )
    parser.add_argument(
        "--eval-id",
        required=True,
        help="ID of the eval to run (created by create_eval.py)",
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=f"Path to a JSONL dataset file (default: {_DEFAULT_DATASET})",
    )
    return parser.parse_args(argv)


async def _collect_outputs(dataset_path: str) -> list[dict]:
    from openai_prep import WorkflowInput, run_workflow  # noqa: PLC0415

    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = await run_workflow(WorkflowInput(input_as_text=item["input_text"]))
            rows.append({"item": item, "sample": {"output_text": result["output_text"]}})
    return rows


async def _run(argv: list[str] | None = None) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args(argv)

    if not os.path.exists(args.dataset):
        print(f"Error: dataset file not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI()

    rows = await _collect_outputs(args.dataset)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="evals")
        run = client.evals.runs.create(
            args.eval_id,
            name="health-workflow-quality",
            data_source={
                "type": "jsonl",
                "source": {"type": "file_id", "id": uploaded.id},
            },
        )
        print(f"Run ID: {run.id}")
    finally:
        os.unlink(tmp_path)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_run(argv))


if __name__ == "__main__":
    main()
