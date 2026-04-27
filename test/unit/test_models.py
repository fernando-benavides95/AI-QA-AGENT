"""
Layer 1 — Unit tests for Pydantic schemas (app/models.py)

VALUE: These schemas are the contract between your code and the LLM.
When with_structured_output() is used, LangChain instructs the LLM to
return JSON conforming to the schema — but the LLM can still violate it.
Pydantic enforces the contract at parse time and raises ValidationError if
the LLM's output doesn't comply.

Testing the schemas means testing that contract is correctly defined.
If a Literal field accepts values it shouldn't, bad LLM output slips through
silently. These tests are your first line of defence before any quality
evaluation happens.
"""
import pytest
from pydantic import ValidationError
from app.models import TestCase, TestSuite, CriticResponse

pytestmark = pytest.mark.unit

VALID_TEST_CASE = {
    "id": "TC01",
    "description": "Successful login with valid credentials",
    "input_data": "email: user@example.com, password: ValidPass1",
    "expected_output": "Redirect to dashboard",
    "priority": "High",
    "category": "Positive",
}


class TestTestCaseSchema:
    # TestCase is the atomic unit of LLM output. Every field the LLM must
    # produce is defined here. The Literal types for priority and category are
    # intentional constraints — they define the vocabulary the LLM must use.

    def test_valid_test_case_constructs(self):
        # Intent: a fully valid TestCase raises no ValidationError.
        TestCase(**VALID_TEST_CASE)

    def test_invalid_priority_raises_validation_error(self):
        # LLMs sometimes return synonyms like "Critical" or "P1".
        # Literal["High", "Medium", "Low"] rejects these at parse time.
        data = {**VALID_TEST_CASE, "priority": "Critical"}
        with pytest.raises(ValidationError):
            TestCase(**data)

    def test_invalid_category_raises_validation_error(self):
        # LLMs may invent categories like "Security" or "Functional".
        # This ensures only the three defined categories are accepted.
        data = {**VALID_TEST_CASE, "category": "Security"}
        with pytest.raises(ValidationError):
            TestCase(**data)

    def test_missing_required_field_raises_validation_error(self):
        data = {k: v for k, v in VALID_TEST_CASE.items() if k != "expected_output"}
        with pytest.raises(ValidationError):
            TestCase(**data)

    def test_unexpected_fields_raise_validation_error(self):
        # LLMs can hallucinate fields not in the schema (e.g. confidence_score,
        # test_type). Without extra="forbid" these are silently dropped, masking
        # schema drift. This ensures any deviation from the contract is caught.
        data = {**VALID_TEST_CASE, "confidence_score": 0.95}
        with pytest.raises(ValidationError):
            TestCase(**data)


class TestTestSuiteSchema:

    def test_valid_suite_with_multiple_cases(self):
        second_case = {**VALID_TEST_CASE, "id": "TC02", "category": "Negative"}
        suite = TestSuite(test_cases=[VALID_TEST_CASE, second_case])
        assert len(suite.test_cases) == 2

    def test_empty_test_cases_list_is_valid(self):
        # The schema permits an empty list — intent here is that no ValidationError is raised.
        TestSuite(test_cases=[])

    def test_additional_considerations_defaults_to_empty_list(self):
        # This field has a default so the LLM can omit it. Confirm the default is an empty list
        suite = TestSuite(test_cases=[])
        assert suite.additional_considerations == []


class TestCriticResponseSchema:
    # CriticResponse is the structured output of the critic agent.
    # The approved field drives the conditional edge in the LangGraph graph —
    # a schema bug here would silently break routing.

    def test_approved_response_without_verified_categories_raises_validation_error(self):
        # An approval with no verified_categories is meaningless — the critic
        # signed off without saying what it checked. Enforce it at schema level.
        with pytest.raises(ValidationError):
            CriticResponse(gaps_identified=[], approved=True, verified_categories=[])

    def test_rejected_response_without_feedback_raises_validation_error(self):
        # If not approved, feedback is required — the generator has no way to
        # improve without it. This is enforced by a model validator in models.py.
        with pytest.raises(ValidationError):
            CriticResponse(gaps_identified=["Missing edge case"], approved=False, feedback="")

    def test_feedback_defaults_to_empty_string(self):
        # When approved, the LLM may omit feedback. Confirm the default
        # is an empty string so callers can safely check truthiness.
        response = CriticResponse(
            gaps_identified=[],
            approved=True,
            verified_categories=["happy path", "input validation"],
        )
        assert response.feedback == ""
