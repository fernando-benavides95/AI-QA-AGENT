from pydantic import BaseModel, Field
from typing import List, Literal


class TestCase(BaseModel):
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
    approved: bool = Field(description="True if the test suite is adequate for the feature's complexity")
    verified_categories: List[str] = Field(
        default_factory=list,
        description="Coverage categories confirmed present (e.g. happy path, auth failures, input validation)"
    )
    feedback: str = Field(
        default="",
        description="Concise actionable improvements if not approved; empty string if approved"
    )
