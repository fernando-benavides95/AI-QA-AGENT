"""
Layer 1 — Unit tests for agent logic in app/agents.py

Three groups of tests, each requiring a different approach:

1. Formatting helpers — pure functions, no mocking needed.
2. Generator state transformation — verifies that generate_test_cases correctly
   updates AgentState fields after an LLM call.
3. Critic state transformation — verifies feedback_history accumulation on
   rejection and status transition on approval.

For groups 2 and 3 the LLM chain is replaced with a MagicMock that returns a
fixed Pydantic object. This isolates the state transformation logic from the
LLM entirely — no API calls, no cost, deterministic results.
"""
import pytest
from unittest.mock import MagicMock
from app.agents import GeneratorAgent, CriticAgent
from app.models import TestSuite, CriticResponse

pytestmark = pytest.mark.unit

VALID_TEST_CASE = {
    "id": "TC01",
    "description": "Successful login",
    "input_data": "email: user@example.com, password: ValidPass1",
    "expected_output": "Redirect to dashboard",
    "priority": "High",
    "category": "Positive",
}


# ---------------------------------------------------------------------------
# Group 1 — Formatting helpers (pure functions)
# ---------------------------------------------------------------------------
#generator:
class TestFormatFeedback:
    # _format_feedback transforms feedback_history into the prompt string
    # the generator receives. If this is wrong, the generator either sees
    # no feedback or sees it in the wrong format — both break the loop.

    def test_empty_history_returns_no_additional_requirements(self):
        agent = GeneratorAgent()
        result = agent._format_feedback([])
        assert result == "No additional requirements."

    def test_single_feedback_entry_rendered_as_bullet_point(self):
        agent = GeneratorAgent()
        result = agent._format_feedback(["Add edge cases."])
        assert "- Add edge cases." in result

    def test_all_feedback_entries_rendered_as_bullet_points(self):
        agent = GeneratorAgent()
        feedback_history = ["Add edge cases.", "Cover SQL injection.", "Add boundary test."]
        result = agent._format_feedback(feedback_history)
        assert "- Add edge cases." in result
        assert "- Cover SQL injection." in result
        assert "- Add boundary test." in result


#critic
class TestFormatForReview:
    # _format_for_review serialises structured test cases into the text the
    # critic LLM receives. Formatting bugs here would cause the critic to
    # review garbled input and produce unreliable feedback.

    def test_all_test_case_fields_present_in_critic_prompt(self):
        # _format_for_review renders structured dicts into human-readable text
        # for the critic's prompt. Every field must survive the transformation —
        # the critic needs input_data and expected_output to judge quality.
        agent = CriticAgent()
        result = agent._format_for_review([VALID_TEST_CASE])
        assert "TC01" in result                                           # id
        assert "Successful login" in result                               # description
        assert "High" in result                                           # priority
        assert "Positive" in result                                       # category
        assert "email: user@example.com, password: ValidPass1" in result  # input_data
        assert "Redirect to dashboard" in result                          # expected_output

    def test_empty_list_returns_empty_string(self):
        agent = CriticAgent()
        assert agent._format_for_review([]) == ""

    def test_multiple_test_cases_each_on_own_line(self):
        second = {**VALID_TEST_CASE, "id": "TC02", "description": "Failed login"}
        agent = CriticAgent()
        result = agent._format_for_review([VALID_TEST_CASE, second])
        lines = result.strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Group 2 — Generator state transformation (mocked chain)
# ---------------------------------------------------------------------------

class TestGeneratorStateTransformation:

    @pytest.fixture
    def generator_with_mock(self, initial_state):
        agent = GeneratorAgent()
        agent.chain = MagicMock()
        agent.chain.invoke.return_value = TestSuite(
            test_cases=[VALID_TEST_CASE],
            additional_considerations=["Check HTTPS"],
        )
        return agent, initial_state

    def test_iterations_increments_by_one(self, generator_with_mock):
        agent, state = generator_with_mock
        result = agent.generate_test_cases(state)
        assert result["iterations"] == state["iterations"] + 1

    def test_status_is_in_progress(self, generator_with_mock):
        agent, state = generator_with_mock
        result = agent.generate_test_cases(state)
        assert result["status"] == "in_progress"

    def test_test_cases_are_serialised_to_dicts(self, generator_with_mock):
        agent, state = generator_with_mock
        result = agent.generate_test_cases(state)
        assert isinstance(result["test_cases"], list)
        assert isinstance(result["test_cases"][0], dict)

    def test_test_cases_contain_required_keys(self, generator_with_mock):
        agent, state = generator_with_mock
        result = agent.generate_test_cases(state)
        tc = result["test_cases"][0]
        for key in ("id", "description", "input_data", "expected_output", "priority", "category"):
            assert key in tc


# ---------------------------------------------------------------------------
# Group 3 — Critic state transformation (mocked chain)
# ---------------------------------------------------------------------------

class TestCriticStateTransformation:

    def _make_critic(self, approved: bool, feedback: str = "", gaps: list = None):
        agent = CriticAgent()
        agent.chain = MagicMock()
        agent.chain.invoke.return_value = CriticResponse(
            gaps_identified=gaps or [],
            approved=approved,
            verified_categories=["happy path"] if approved else [],
            feedback=feedback,
        )
        return agent

    def test_rejection_appends_feedback_to_history(self, initial_state):
        agent = self._make_critic(approved=False, feedback="Add edge cases.", gaps=["Missing edge case"])
        result = agent.review_test_cases(initial_state)
        assert "Add edge cases." in result["feedback_history"]

    def test_rejection_preserves_existing_history(self, initial_state):
        state = {**initial_state, "feedback_history": ["Round 1 feedback."]}
        agent = self._make_critic(approved=False, feedback="Add boundary tests.", gaps=["Missing boundary"])
        result = agent.review_test_cases(state)
        assert len(result["feedback_history"]) == 2
        assert result["feedback_history"][0] == "Round 1 feedback."
        assert result["feedback_history"][1] == "Add boundary tests."

    def test_rejection_status_remains_in_progress(self, initial_state):
        agent = self._make_critic(approved=False, feedback="Add edge cases.", gaps=["gap"])
        result = agent.review_test_cases(initial_state)
        assert result["status"] == "in_progress"

    def test_approval_sets_status_to_approved(self, initial_state):
        agent = self._make_critic(approved=True)
        result = agent.review_test_cases(initial_state)
        assert result["status"] == "approved"

    def test_approval_does_not_modify_feedback_history(self, initial_state):
        state = {**initial_state, "feedback_history": ["some prior feedback"]}
        agent = self._make_critic(approved=True)
        result = agent.review_test_cases(state)
        assert result["feedback_history"] == ["some prior feedback"]
