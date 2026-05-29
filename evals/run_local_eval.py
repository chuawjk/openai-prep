"""Local eval runner for the health-coach workflow.

Usage:
    uv run python evals/run_local_evals.py [--dataset PATH]

For each item in the dataset:
1. Runs the agent workflow to produce output_text.
2. Grades output_text against eval_guide (pass/fail + one-line reason).
3. Writes results to evals/results/<timestamp>/eval_results.csv.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from agents.tracing.spans import Span
    from agents.tracing.traces import Trace
    from openai import AsyncOpenAI

_DEFAULT_DATASET = "evals/data/health_queries.jsonl"
_RESULTS_DIR = "evals/results"
_GRADER_MODEL = "gpt-4.1"

_GRADER_SYSTEM_PROMPT = (
    "You are an evaluator for a health-coach chatbot. "
    "Given a user query, the chatbot's response, and an evaluation criterion, "
    "decide whether the response passes or fails the criterion. "
    "Return a structured result with eval_outcome (pass or fail) and a concise one-line reason."
)


class _EvalTraceCollector:
    """TracingProcessor that accumulates spans per trace so each eval example
    can retrieve its own completed trace record after run_workflow returns."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, list[dict[str, Any]]] = {}  # trace_id -> spans
        self._completed: list[dict[str, Any]] = []

    # ---- TracingProcessor interface ----

    def on_trace_start(self, trace: "Trace") -> None:
        with self._lock:
            self._active[trace.trace_id] = []

    def on_trace_end(self, trace: "Trace") -> None:
        with self._lock:
            spans = self._active.pop(trace.trace_id, [])
            self._completed.append({
                "trace_id": trace.trace_id,
                "name": trace.name,
                "spans": spans,
            })

    def on_span_start(self, span: "Span[Any]") -> None:
        pass

    def on_span_end(self, span: "Span[Any]") -> None:
        with self._lock:
            bucket = self._active.get(span.trace_id)
            if bucket is None:
                return

            from agents.tracing.span_data import ResponseSpanData  # noqa: PLC0415

            span_data = span.span_data
            exported = span_data.export()

            # ResponseSpanData.export() intentionally omits input/output content
            # (it only keeps response_id + usage). Access the raw attributes directly
            # so we capture what was sent to and received from each agent.
            if isinstance(span_data, ResponseSpanData):
                if span_data.input is not None:
                    exported["input"] = span_data.input
                if span_data.response is not None:
                    try:
                        output_items = [
                            item.model_dump() for item in span_data.response.output
                        ]
                        exported["output"] = output_items
                        # Extract tool calls into a dedicated field so arguments are
                        # immediately visible without digging into the full output list.
                        tool_calls = []
                        for item in output_items:
                            item_type = item.get("type", "")
                            if item_type == "function_call":
                                args = item.get("arguments", "")
                                try:
                                    args = json.loads(args)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                                tool_calls.append({
                                    "type": item_type,
                                    "name": item.get("name"),
                                    "arguments": args,
                                    "call_id": item.get("call_id"),
                                })
                            elif item_type == "web_search_call":
                                tool_calls.append({
                                    "type": item_type,
                                    "queries": item.get("queries", []),
                                    "call_id": item.get("id"),
                                })
                        if tool_calls:
                            exported["tool_calls"] = tool_calls
                    except Exception:
                        exported["output"] = str(span_data.response)

            record: dict[str, Any] = {
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "started_at": span.started_at,
                "ended_at": span.ended_at,
                **exported,
            }
            if span.error:
                record["error"] = span.error
            bucket.append(record)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # ---- Retrieval ----

    def pop_last_completed(self) -> dict[str, Any] | None:
        """Return and remove the most recently completed trace record."""
        with self._lock:
            return self._completed.pop() if self._completed else None


def _build_span_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the flat span list into a nested tree sorted by start time.

    Spans arrive in bottom-up completion order (inner spans end before outer
    ones). This function re-roots them so the output reads top-down,
    chronologically, matching the actual execution sequence.
    """
    # Attach a temporary _children accumulator to every span.
    by_id: dict[str, dict[str, Any]] = {
        s["span_id"]: {**s, "_children": []} for s in spans
    }

    roots: list[dict[str, Any]] = []
    for span in by_id.values():
        parent_id = span.get("parent_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["_children"].append(span)
        else:
            roots.append(span)

    def _finalise(node: dict[str, Any]) -> dict[str, Any]:
        children = sorted(
            node.pop("_children"), key=lambda s: s.get("started_at") or ""
        )
        out = dict(node)
        if children:
            out["children"] = [_finalise(c) for c in children]
        return out

    roots.sort(key=lambda s: s.get("started_at") or "")
    return [_finalise(r) for r in roots]


class GradeResult(BaseModel):
    eval_outcome: Literal["pass", "fail"]
    reason: str


async def _grade(
    client: "AsyncOpenAI",
    input_text: str,
    output_text: str,
    eval_guide: str,
) -> GradeResult:
    response = await client.beta.chat.completions.parse(
        model=_GRADER_MODEL,
        messages=[
            {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User query: {input_text}\n\n"
                    f"Chatbot response:\n{output_text}\n\n"
                    f"Evaluation criterion: {eval_guide}"
                ),
            },
        ],
        response_format=GradeResult,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Grader returned no structured output")
    return parsed


async def _run_eval(dataset_path: str) -> None:
    import tqdm  # noqa: PLC0415

    from openai import AsyncOpenAI  # noqa: PLC0415
    from agents.tracing.setup import get_trace_provider  # noqa: PLC0415

    from openai_prep import WorkflowInput, run_workflow  # noqa: PLC0415

    client = AsyncOpenAI()

    collector = _EvalTraceCollector()
    # _EvalTraceCollector satisfies the TracingProcessor interface; register it
    # so the SDK calls our hooks for every trace/span produced by run_workflow.
    get_trace_provider().register_processor(collector)  # type: ignore[arg-type]

    items: list[dict] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        print("Error: dataset is empty.", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(_RESULTS_DIR) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.csv"

    fieldnames = list(items[0].keys()) + ["output_text", "eval_outcome", "reason"]
    traces_path = out_dir / "traces.jsonl"

    passed = 0
    with (
        open(out_path, "w", newline="", encoding="utf-8") as csv_file,
        open(traces_path, "w", encoding="utf-8") as traces_file,
        tqdm.tqdm(total=len(items), unit="example", dynamic_ncols=True) as pbar,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            pbar.set_description(item["eval_id"])

            try:
                workflow_result = await run_workflow(
                    WorkflowInput(input_as_text=item["input_text"])
                )
                output_text = workflow_result["output_text"]
            except Exception as exc:
                row = {**item, "output_text": "", "eval_outcome": "fail", "reason": f"Workflow error: {exc}"}
                writer.writerow(row)
                csv_file.flush()
                pbar.set_postfix(outcome="fail")
                pbar.update(1)
                continue
            finally:
                trace_record = collector.pop_last_completed()
                if trace_record:
                    trace_record["eval_id"] = item["eval_id"]
                    trace_record["spans"] = _build_span_tree(trace_record["spans"])
                    traces_file.write(json.dumps(trace_record) + "\n")
                    traces_file.flush()

            try:
                grade = await _grade(client, item["input_text"], output_text, item.get("eval_guide", ""))
            except Exception as exc:
                row = {**item, "output_text": output_text, "eval_outcome": "fail", "reason": f"Grader error: {exc}"}
                writer.writerow(row)
                csv_file.flush()
                pbar.set_postfix(outcome="fail")
                pbar.update(1)
                continue

            row = {**item, "output_text": output_text, "eval_outcome": grade.eval_outcome, "reason": grade.reason}
            writer.writerow(row)
            csv_file.flush()

            if grade.eval_outcome == "pass":
                passed += 1
            pbar.set_postfix(outcome=grade.eval_outcome)
            pbar.update(1)

    print(f"\nResults: {passed}/{len(items)} passed")
    print(f"CSV:    {out_path}")
    print(f"Traces: {traces_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local evals for the health-coach workflow."
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=f"Path to JSONL dataset (default: {_DEFAULT_DATASET})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set.\n  export OPENAI_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run_eval(args.dataset))


if __name__ == "__main__":
    main()
