"""Tests for the live verification command."""

from unittest.mock import AsyncMock, patch

import pytest

import openai_prep.verify as verify_mod


@pytest.mark.anyio
async def test_missing_api_key_exits_with_code_1(monkeypatch, capsys):
    """verify.main() exits with code 1 and prints a helpful message when OPENAI_API_KEY is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        await verify_mod.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err


@pytest.mark.anyio
async def test_empty_api_key_exits_with_code_1(monkeypatch, capsys):
    """verify.main() exits with code 1 and prints a helpful message when OPENAI_API_KEY is empty."""
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(SystemExit) as exc_info:
        await verify_mod.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err


@pytest.mark.anyio
async def test_valid_api_key_calls_run_workflow(monkeypatch, capsys):
    """verify.main() calls run_workflow and prints output_text when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    mock_run_workflow = AsyncMock(return_value={"output_text": "cardio is good"})

    with patch("openai_prep.run_workflow", mock_run_workflow):
        await verify_mod.main()

    mock_run_workflow.assert_awaited_once()
    captured = capsys.readouterr()
    assert "cardio is good" in captured.out
