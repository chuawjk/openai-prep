import pytest
from pydantic import ValidationError

from openai_prep.schemas import (
    InformationAgentSchema,
    InformationAgentSchema__Sources,
    OrchestratorAgentSchema,
    RecommenderAgentSchema,
    WorkflowInput,
)


def test_workflow_input_constructs_and_dumps():
    model = WorkflowInput(input_as_text="I want a light workout.")

    assert model.input_as_text == "I want a light workout."
    assert model.model_dump() == {"input_as_text": "I want a light workout."}


def test_orchestrator_schema_constructs_and_dumps():
    model = OrchestratorAgentSchema(category="Recommendation")

    assert model.category == "Recommendation"
    assert model.model_dump() == {"category": "Recommendation"}


def test_recommender_schema_constructs_and_dumps():
    model = RecommenderAgentSchema(
        activity="Walking",
        intensity="Light",
        frequency="30 minutes daily",
        justification="Low-impact movement supports general health.",
    )

    assert model.model_dump() == {
        "activity": "Walking",
        "intensity": "Light",
        "frequency": "30 minutes daily",
        "justification": "Low-impact movement supports general health.",
    }


def test_information_schema_constructs_nested_sources_and_dumps():
    model = InformationAgentSchema(
        information="Hydration supports normal body function.",
        sources={
            "url_1": "https://healthhub.sg/a",
            "url_2": "https://healthhub.sg/b",
            "url_3": "https://healthhub.sg/c",
        },
    )

    assert isinstance(model.sources, InformationAgentSchema__Sources)
    assert model.sources.url_1 == "https://healthhub.sg/a"
    assert model.model_dump() == {
        "information": "Hydration supports normal body function.",
        "sources": {
            "url_1": "https://healthhub.sg/a",
            "url_2": "https://healthhub.sg/b",
            "url_3": "https://healthhub.sg/c",
        },
    }


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (WorkflowInput, {}),
        (OrchestratorAgentSchema, {}),
        (
            RecommenderAgentSchema,
            {
                "activity": "Walking",
                "intensity": "Light",
                "frequency": "30 minutes daily",
            },
        ),
        (
            InformationAgentSchema__Sources,
            {
                "url_1": "https://healthhub.sg/a",
                "url_2": "https://healthhub.sg/b",
            },
        ),
        (
            InformationAgentSchema,
            {
                "information": "Hydration supports normal body function.",
            },
        ),
    ],
)
def test_schema_validation_fails_for_missing_required_fields(schema, payload):
    with pytest.raises(ValidationError):
        schema(**payload)


def test_orchestrator_schema_rejects_unknown_category():
    with pytest.raises(ValidationError):
        OrchestratorAgentSchema(category="Other")
