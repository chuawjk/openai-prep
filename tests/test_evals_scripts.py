"""Tests for evals/create_eval.py and evals/run_eval.py.

These tests are purely unit/env-guard tests — no live API calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add evals/ to sys.path so we can import the scripts directly
_EVALS_DIR = str(Path(__file__).parent.parent / "evals")
if _EVALS_DIR not in sys.path:
    sys.path.insert(0, _EVALS_DIR)


# ---------------------------------------------------------------------------
# create_eval.py tests
# ---------------------------------------------------------------------------


class TestCreateEvalEnvGuard:
    """create_eval.main() must fast-fail when OPENAI_API_KEY is missing or empty."""

    def test_missing_api_key_exits_with_code_1(self, monkeypatch, capsys):
        """Exits with code 1 and prints OPENAI_API_KEY to stderr when key is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        import create_eval  # noqa: PLC0415

        with pytest.raises(SystemExit) as exc_info:
            create_eval.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.err

    def test_empty_api_key_exits_with_code_1(self, monkeypatch, capsys):
        """Exits with code 1 and prints OPENAI_API_KEY to stderr when key is empty."""
        monkeypatch.setenv("OPENAI_API_KEY", "")

        import create_eval  # noqa: PLC0415

        with pytest.raises(SystemExit) as exc_info:
            create_eval.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.err


class TestCreateEvalArgParsing:
    """parse_args() must accept --eval-name and fall back to a default."""

    def test_default_eval_name(self):
        """parse_args([]) returns a default eval_name."""
        import create_eval  # noqa: PLC0415

        args = create_eval.parse_args([])
        assert args.eval_name == "Health Query Routing"

    def test_custom_eval_name(self):
        """parse_args(['--eval-name', 'My Eval']) sets eval_name correctly."""
        import create_eval  # noqa: PLC0415

        args = create_eval.parse_args(["--eval-name", "My Eval"])
        assert args.eval_name == "My Eval"


class TestCreateEvalCallsAPI:
    """create_eval.main() must call client.evals.create with correct arguments."""

    def test_calls_evals_create_with_string_check_grader(self, monkeypatch, capsys):
        """When OPENAI_API_KEY is set, main() calls client.evals.create once."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        mock_eval = MagicMock()
        mock_eval.id = "eval_test_123"

        mock_client = MagicMock()
        mock_client.evals.create.return_value = mock_eval

        import create_eval  # noqa: PLC0415

        with patch("openai.OpenAI", return_value=mock_client):
            create_eval.main([])

        mock_client.evals.create.assert_called_once()
        call_kwargs = mock_client.evals.create.call_args[1]

        # Verify testing_criteria contains a string_check grader
        criteria = call_kwargs["testing_criteria"]
        assert len(criteria) == 1
        assert criteria[0]["type"] == "string_check"
        assert criteria[0]["operation"] == "eq"

        # Verify eval id is printed
        captured = capsys.readouterr()
        assert "eval_test_123" in captured.out


# ---------------------------------------------------------------------------
# run_eval.py tests
# ---------------------------------------------------------------------------


class TestRunEvalEnvGuard:
    """run_eval.main() must fast-fail when OPENAI_API_KEY is missing or empty."""

    def test_missing_api_key_exits_with_code_1(self, monkeypatch, capsys):
        """Exits with code 1 and prints OPENAI_API_KEY to stderr when key is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        import run_eval  # noqa: PLC0415

        with pytest.raises(SystemExit) as exc_info:
            run_eval.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.err

    def test_empty_api_key_exits_with_code_1(self, monkeypatch, capsys):
        """Exits with code 1 and prints OPENAI_API_KEY to stderr when key is empty."""
        monkeypatch.setenv("OPENAI_API_KEY", "")

        import run_eval  # noqa: PLC0415

        with pytest.raises(SystemExit) as exc_info:
            run_eval.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.err


class TestRunEvalArgParsing:
    """parse_args() must require --eval-id and accept optional --dataset."""

    def test_eval_id_is_required(self):
        """parse_args([]) raises SystemExit because --eval-id is required."""
        import run_eval  # noqa: PLC0415

        with pytest.raises(SystemExit):
            run_eval.parse_args([])

    def test_eval_id_is_parsed(self):
        """parse_args(['--eval-id', 'eval_abc']) sets eval_id correctly."""
        import run_eval  # noqa: PLC0415

        args = run_eval.parse_args(["--eval-id", "eval_abc"])
        assert args.eval_id == "eval_abc"

    def test_default_dataset_path(self):
        """parse_args with only --eval-id uses the bundled dataset path."""
        import run_eval  # noqa: PLC0415

        args = run_eval.parse_args(["--eval-id", "eval_abc"])
        assert "health_queries.jsonl" in args.dataset

    def test_custom_dataset_path(self):
        """parse_args accepts a custom --dataset path."""
        import run_eval  # noqa: PLC0415

        args = run_eval.parse_args(["--eval-id", "eval_abc", "--dataset", "/tmp/test.jsonl"])
        assert args.dataset == "/tmp/test.jsonl"


class TestRunEvalMissingDataset:
    """run_eval exits with code 1 when the dataset file does not exist."""

    def test_missing_dataset_exits_with_code_1(self, monkeypatch, capsys, tmp_path):
        """Exits with code 1 when dataset path does not exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        import run_eval  # noqa: PLC0415

        missing_path = str(tmp_path / "nonexistent.jsonl")

        with pytest.raises(SystemExit) as exc_info:
            run_eval.main(["--eval-id", "eval_abc", "--dataset", missing_path])

        assert exc_info.value.code == 1
