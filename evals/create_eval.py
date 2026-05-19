"""Create an OpenAI eval definition for health query routing.

Usage:
    uv run python evals/create_eval.py [--eval-name NAME]

Prints the created eval ID to stdout.
Requires OPENAI_API_KEY to be set in the environment.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an OpenAI eval definition for health query routing."
    )
    parser.add_argument(
        "--eval-name",
        default="Health Query Routing",
        help="Name of the eval to create (default: 'Health Query Routing')",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
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
    eval_name: str = args.eval_name

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=api_key)

    eval_obj = client.evals.create(
        name=eval_name,
        data_source_config={
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "input_text": {"type": "string"},
                    "expected_category": {"type": "string"},
                },
                "required": ["input_text", "expected_category"],
            },
            "include_sample_schema": True,
        },
        testing_criteria=[
            {
                "type": "string_check",
                "name": "Routing correctness",
                "input": "{{ sample.output_text }}",
                "operation": "eq",
                "reference": "{{ item.expected_category }}",
            }
        ],
    )

    print(eval_obj.id)


if __name__ == "__main__":
    main()
