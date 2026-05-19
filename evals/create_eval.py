"""Create an OpenAI eval definition for health workflow output quality.

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
        description="Create an OpenAI eval definition for health workflow output quality."
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
                },
                "required": ["input_text"],
            },
            "include_sample_schema": False,
        },
        testing_criteria=[
            {
                "type": "label_model",
                "name": "Relevance",
                "model": "gpt-4o-mini",
                "input": [
                    {
                        "role": "system",
                        "content": (
                            "You are evaluating a health assistant response. "
                            "A response is RELEVANT if it meaningfully addresses the user's query — "
                            "including a polite refusal when the query is harmful or out of scope."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Query: {{ item.input_text }}\n\nResponse: {{ sample.output_text }}\n\nIs this response relevant to the query?",
                    },
                ],
                "labels": ["relevant", "not_relevant"],
                "passing_labels": ["relevant"],
            },
            {
                "type": "label_model",
                "name": "Politeness",
                "model": "gpt-4o-mini",
                "input": [
                    {
                        "role": "system",
                        "content": "You are evaluating the tone of a health assistant response.",
                    },
                    {
                        "role": "user",
                        "content": "Response: {{ sample.output_text }}\n\nIs this response polite and appropriate in tone?",
                    },
                ],
                "labels": ["polite", "impolite"],
                "passing_labels": ["polite"],
            },
        ],
    )

    print(eval_obj.id)


if __name__ == "__main__":
    main()
