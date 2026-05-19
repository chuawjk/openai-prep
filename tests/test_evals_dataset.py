"""Tests for the evals dataset file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

_DATASET_PATH = Path(__file__).parent.parent / "evals" / "data" / "health_queries.jsonl"

_VALID_CATEGORIES = {"Information", "Recommendation", "Rejection"}
_MIN_ROWS_PER_CATEGORY = 3


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


def test_dataset_has_required_fields():
    """Every row must have 'input_text' and 'expected_category' fields."""
    rows = _load_dataset()
    for i, row in enumerate(rows, start=1):
        assert "input_text" in row, f"Row {i} missing 'input_text'"
        assert "expected_category" in row, f"Row {i} missing 'expected_category'"
        assert isinstance(row["input_text"], str), (
            f"Row {i}: 'input_text' must be a string"
        )
        assert isinstance(row["expected_category"], str), (
            f"Row {i}: 'expected_category' must be a string"
        )


def test_dataset_categories_are_valid():
    """All 'expected_category' values must be one of Information, Recommendation, Rejection."""
    rows = _load_dataset()
    for i, row in enumerate(rows, start=1):
        cat = row["expected_category"]
        assert cat in _VALID_CATEGORIES, (
            f"Row {i}: unexpected category '{cat}'. "
            f"Must be one of {_VALID_CATEGORIES}"
        )


def test_dataset_has_all_three_categories():
    """The dataset must include at least one row for each routing category."""
    rows = _load_dataset()
    categories_present = {row["expected_category"] for row in rows}
    missing = _VALID_CATEGORIES - categories_present
    assert not missing, f"Dataset missing categories: {missing}"


def test_dataset_has_at_least_three_rows_per_category():
    """Each category must have at least 3 rows (not just 1)."""
    rows = _load_dataset()
    counts = Counter(row["expected_category"] for row in rows)
    for category in _VALID_CATEGORIES:
        actual = counts.get(category, 0)
        assert actual >= _MIN_ROWS_PER_CATEGORY, (
            f"Category '{category}' has {actual} row(s); "
            f"expected at least {_MIN_ROWS_PER_CATEGORY}"
        )


def test_dataset_has_minimum_total_rows():
    """The dataset must have at least 9 rows total (3 per category × 3 categories)."""
    rows = _load_dataset()
    assert len(rows) >= 9, (
        f"Dataset has {len(rows)} row(s); expected at least 9"
    )


def test_dataset_input_texts_are_non_empty():
    """Every 'input_text' must be a non-empty string."""
    rows = _load_dataset()
    for i, row in enumerate(rows, start=1):
        assert row["input_text"].strip(), f"Row {i}: 'input_text' is empty"
