# Example entry point for the openai-prep application.
import asyncio

from openai_prep import WorkflowInput, run_workflow


async def main():
    workflow_input = WorkflowInput(input_as_text="What exercise helps cardiovascular health?")
    result = await run_workflow(workflow_input)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
