"""Typed schemas for the OpenAI Prep workflow."""

from typing import Literal

from pydantic import BaseModel


class WorkflowInput(BaseModel):
    input_as_text: str


class OrchestratorAgentSchema(BaseModel):
    category: Literal["Recommendation", "Information"]


class RecommenderAgentSchema(BaseModel):
    activity: str
    intensity: str
    frequency: str
    justification: str


class InformationAgentSchema__Sources(BaseModel):
    url_1: str
    url_2: str
    url_3: str


class InformationAgentSchema(BaseModel):
    information: str
    sources: InformationAgentSchema__Sources
