"""
Layer 2 — Coverage quality rubric (manual LLM-as-judge)

CONCEPT: G-Eval (Generative Evaluation) means using an LLM to score output
against a rubric you define. The rubric encodes what "good" looks like for
your specific system — this is where your QA domain knowledge becomes the test.

HOW IT WORKS: We send the feature, the test cases, and a list of criteria to
the judge. The judge scores each criterion 0 or 1, computes the average, and
returns JSON. We assert the average score >= threshold.

The evaluation_steps list below IS the rubric. Defining good criteria is the
hardest and most transferable skill in AI evaluation — if the rubric is weak,
the score is meaningless regardless of the framework used.

Run with: pytest -m llm
"""
import pytest
from .conftest import EVALUATION_FEATURE

pytestmark = pytest.mark.llm

THRESHOLD = 0.7

RUBRIC_PROMPT = """
You are an expert software QA evaluator.

Evaluate the following test suite against each criterion below.
Score each criterion: 1 if clearly present, 0 if missing or unclear.
Compute the overall score as the average of all criterion scores.

Feature:
{feature}

Generated test cases:
{output}

Criteria:
1. At least one positive test case covers the successful happy path
2. At least one negative test case covers an authentication or error failure
3. At least one test case covers input validation (empty fields, invalid format)
4. At least one edge case or boundary condition is tested
5. At least one security concern is addressed (e.g. SQL injection, brute force)

Respond with ONLY valid JSON — no markdown, no prose outside the JSON object:
{{
  "criterion_scores": [<1 or 0>, <1 or 0>, <1 or 0>, <1 or 0>, <1 or 0>],
  "score": <average of criterion_scores as float>,
  "reason": "<one sentence summarising the evaluation>"
}}
"""


def test_test_suite_meets_coverage_quality_rubric(judge, formatted_output):
    prompt = RUBRIC_PROMPT.format(
        feature=EVALUATION_FEATURE,
        output=formatted_output,
    )
    result = judge.evaluate(prompt)
    score = result["score"]
    criteria = result.get("criterion_scores", [])
    reason = result.get("reason", "no reason returned")

    criteria_labels = [
        "Happy path",
        "Auth/error failure",
        "Input validation",
        "Edge/boundary case",
        "Security",
    ]
    print(f"\nCoverage Rubric:  {score:.2f} / 1.0  (threshold: {THRESHOLD})")
    for label, s in zip(criteria_labels, criteria):
        print(f"  {'[pass]' if s else '[FAIL]'} {label}")
    print(f"Reason: {reason}")

    assert score >= THRESHOLD, (
        f"Rubric score {score:.2f} is below threshold {THRESHOLD}\n"
        f"Criterion scores: {dict(zip(criteria_labels, criteria))}\n"
        f"Reason: {reason}"
    )
