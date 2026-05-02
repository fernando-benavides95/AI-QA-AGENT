"""
Layer 2 — G-Eval with custom rubric (DeepEval)

CONCEPT: G-Eval lets you define what "good" means as a scoring rubric.
The judge LLM evaluates the output against each criterion and produces a
composite score. This is where your QA domain knowledge becomes the test —
you are writing the acceptance criteria for LLM output quality.

Unlike Answer Relevancy (which is generic), G-Eval is specific to your
system. These criteria encode what a QA engineer would check when reviewing
a test suite. If the rubric is wrong, the score is wrong — defining good
criteria is the hardest and most valuable skill in AI evaluation.

Run with: pytest -m evaluation
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from .conftest import EVALUATION_FEATURE

pytestmark = [pytest.mark.llm, pytest.mark.evaluation]


def test_test_suite_meets_coverage_quality_rubric(gemini_judge, formatted_output):
    """
    Scores the generated test suite against a QA coverage rubric.
    Each criterion below is something a senior QA engineer would verify
    manually — G-Eval automates that review.
    """
    test_case = LLMTestCase(
        input=EVALUATION_FEATURE,
        actual_output=formatted_output,
    )

    metric = GEval(
        name="Test Suite Coverage Quality",
        criteria=(
            "Evaluate whether the test suite provides comprehensive and appropriate "
            "coverage for the described software feature."
        ),
        evaluation_steps=[
            "Verify at least one positive test case covers the successful happy path.",
            "Verify at least one negative test case covers an authentication failure scenario.",
            "Verify at least one test case covers input validation (e.g. empty fields, invalid format).",
            "Verify at least one edge case or boundary condition is tested.",
            "Verify at least one security concern is addressed (e.g. SQL injection, brute force).",
            "Verify there are no semantically redundant test cases — two test cases that test "
            "the same scenario with different wording count as duplicates and reduce suite quality.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        model=gemini_judge,
    )
    assert_test(test_case, [metric])
