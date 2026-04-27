"""
Layer 2 — Answer Relevancy evaluation (DeepEval)

CONCEPT: Answer Relevancy checks whether the output is actually addressing
the input. For our system: do the generated test cases relate to the feature
described, or has the agent produced generic/hallucinated cases unrelated to
the feature?

This is the most fundamental quality gate for an LLM output — before asking
"is it good?", ask "is it even about the right thing?".

LLM-AS-JUDGE: DeepEval sends both the input and output to a judge LLM
(GeminiJudge) which scores the relevancy 0.0-1.0. The assertion is
probabilistic: score >= threshold, not output == expected.

Run with: pytest -m llm
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from .conftest import EVALUATION_FEATURE

pytestmark = pytest.mark.llm


def test_generated_cases_are_relevant_to_feature(gemini_judge, formatted_output):
    """
    Asserts that the generated test cases address the feature description.
    A score below 0.7 means the agent produced off-topic or hallucinated cases.
    """
    test_case = LLMTestCase(
        input=EVALUATION_FEATURE,
        actual_output=formatted_output,
    )
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True,  # judge explains its score — useful for debugging
    )
    assert_test(test_case, [metric])
