# Example entry point for the openai-prep application.
import asyncio

from openai_prep import WorkflowInput, run_workflow

_QUERIES = [
    # Information path: factual question expects orchestrator category "Information"
    "What are the health benefits of regular exercise?",
    # Recommendation path: advice-seeking expects orchestrator category "Recommendation"
    "I have been feeling tired lately and want to improve my energy levels. What should I do?",
    # This should trigger guardrails
    "But I am very poorly today & very stupid & hate everybody & everything."
]


async def main():
    for query in _QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        workflow_input = WorkflowInput(input_as_text=query)
        result = await run_workflow(workflow_input)
        print(result["output_text"])


if __name__ == "__main__":
    asyncio.run(main())
