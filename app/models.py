from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Literal, Optional


class FeatureValidation(BaseModel):
    is_feature: bool = Field(description="True if the input describes a software feature that can be tested")
    feedback: str = Field(description="Brief explanation of why this is or is not a software feature, or what is unclear")


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Test case identifier, e.g. TC01")
    description: str = Field(description="What is being tested")
    input_data: str = Field(description="Input values or preconditions")
    expected_output: str = Field(description="Expected result or system behaviour")
    priority: Literal["High", "Medium", "Low"]
    category: Literal["Positive", "Negative", "Edge Case"]


class TestSuite(BaseModel):
    test_cases: List[TestCase]
    additional_considerations: List[str] = Field(
        default_factory=list,
        description="Non-functional or complementary testing notes (security, performance, accessibility, etc.)"
    )


class CriticResponse(BaseModel):
    gaps_identified: List[str] = Field(
        description=(
            "List every coverage gap found, including minor ones. "
            "For features with multiple business rules, check rule combinations — "
            "not just individual rules in isolation. Must be populated before deciding to approve. "
            "Use an empty list only if you genuinely find no gaps."
        )
    )
    approved: bool = Field(
        description="True only if gaps_identified contains no critical gaps for this feature's complexity"
    )
    verified_categories: List[str] = Field(
        default_factory=list,
        description="Coverage categories confirmed present (e.g. happy path, auth failures, input validation)"
    )
    feedback: str = Field(
        default="",
        description="Concise actionable improvements drawn from gaps_identified; empty string if approved"
    )

    @model_validator(mode="after")
    def validate_approval_fields(self) -> "CriticResponse":
        if not self.approved and not self.feedback.strip():
            raise ValueError("feedback must not be empty when approved is False")
        if self.approved and not self.verified_categories:
            raise ValueError("verified_categories must not be empty when approved is True")
        return self
