"""Tests for the evals dataset file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATASET_PATH = Path(__file__).parent.parent / "evals" / "data" / "health_queries.jsonl"

_MIN_TOTAL_ROWS = 9


def _load_dataset() -> list[dict]:
    """Load and parse all rows from the JSONL dataset."""
    rows: list[dict] = []
    with _DATASET_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_dataset_file_exists():
    """The dataset file must exist at the expected path."""
    assert _DATASET_PATH.exists(), f"Dataset not found: {_DATASET_PATH}"


def test_dataset_is_valid_jsonl():
    """Every non-empty line must be valid JSON."""
    with _DATASET_PATH.open() as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc}")


def test_dataset_has_input_text_field():
    """Every row must have an 'input_text' field that is a non-empty string."""
    rows = _load_dataset()
    for i, row in enumerate(rows, start=1):
        assert "input_text" in row, f"Row {i} missing 'input_text'"
        assert isinstance(row["input_text"], str), (
            f"Row {i}: 'input_text' must be a string"
        )
        assert row["input_text"].strip(), f"Row {i}: 'input_text' is empty"


def test_dataset_has_no_expected_category():
    """No row should have an 'expected_category' field."""
    rows = _load_dataset()
    for i, row in enumerate(rows, start=1):
        assert "expected_category" not in row, (
            f"Row {i} must not contain 'expected_category'"
        )


def test_dataset_has_minimum_total_rows():
    """The dataset must have at least 9 rows total (3 per query type × 3 types)."""
    rows = _load_dataset()
    assert len(rows) >= _MIN_TOTAL_ROWS, (
        f"Dataset has {len(rows)} row(s); expected at least {_MIN_TOTAL_ROWS}"
    )
