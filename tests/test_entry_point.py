"""Tests for the public entry point exposed via the openai_prep package."""

from unittest.mock import AsyncMock

import pytest

import openai_prep
from openai_prep import WorkflowInput, run_workflow


@pytest.mark.anyio
async def test_entry_point_calls_run_workflow_with_expected_args(monkeypatch):
    """Public entry point calls run_workflow with the WorkflowInput and no extra kwargs."""
    workflow_input = WorkflowInput(input_as_text="What exercise helps cardiovascular health?")
    mock_run_workflow = AsyncMock(return_value={"output_text": "test response"})
    monkeypatch.setattr(openai_prep, "run_workflow", mock_run_workflow)

    result = await openai_prep.run_workflow(workflow_input)

    mock_run_workflow.assert_awaited_once_with(workflow_input)
    assert result == {"output_text": "test response"}


@pytest.mark.anyio
async def test_entry_point_returns_expected_output_shape(monkeypatch):
    """Public entry point returns a dict with an 'output_text' key containing a string."""
    workflow_input = WorkflowInput(input_as_text="What exercise helps cardiovascular health?")
    mock_run_workflow = AsyncMock(return_value={"output_text": "test response"})
    monkeypatch.setattr(openai_prep, "run_workflow", mock_run_workflow)

    result = await openai_prep.run_workflow(workflow_input)

    assert isinstance(result, dict)
    assert "output_text" in result
    assert isinstance(result["output_text"], str)
