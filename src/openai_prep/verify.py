"""Live verification command for the openai_prep workflow.

Invokable as: uv run python -m openai_prep.verify
"""

import asyncio
import os
import sys

_SAMPLE_QUERY = "What exercise helps cardiovascular health?"


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your shell before running live verification:\n"
            "  export OPENAI_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    from openai_prep import WorkflowInput, run_workflow

    workflow_input = WorkflowInput(input_as_text=_SAMPLE_QUERY)
    result = await run_workflow(workflow_input)
    print(result["output_text"])


if __name__ == "__main__":
    asyncio.run(main())
