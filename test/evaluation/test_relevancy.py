"""
Layer 2 — Answer Relevancy (manual LLM-as-judge)

CONCEPT: Before asking "are the test cases good?", ask "are they even about
the right feature?". Relevancy catches hallucinated or off-topic output —
the most basic quality failure an LLM can produce.

HOW IT WORKS: We send the feature description and the generated test cases
to a judge LLM with a prompt that asks it to score relevancy 0.0-1.0.
The judge returns JSON. We assert score >= threshold.

The prompt below IS the metric definition. If you want to change what
"relevancy" means for this system, change the prompt — not the code.

Run with: pytest -m llm
"""
import pytest
from .conftest import EVALUATION_FEATURE

pytestmark = pytest.mark.llm

THRESHOLD = 0.7

RELEVANCY_PROMPT = """
You are an expert software testing evaluator.

Given a software feature description and a set of generated test cases, score
how relevant the test cases are to the feature being described.

Feature:
{feature}

Generated test cases:
{output}

Respond with ONLY valid JSON — no markdown, no prose outside the JSON object:
{{"score": <float between 0.0 and 1.0>, "reason": "<one sentence explaining the score>"}}

Scoring guide:
  1.0 — every test case directly addresses the described feature
  0.7 — most test cases are relevant, minor gaps or tangents
  0.5 — mixed relevancy, some cases appear off-topic or hallucinated
  0.0 — test cases do not address the described feature at all
"""


def test_generated_cases_are_relevant_to_feature(judge, formatted_output):
    prompt = RELEVANCY_PROMPT.format(
        feature=EVALUATION_FEATURE,
        output=formatted_output,
    )
    result = judge.evaluate(prompt)
    score = result["score"]
    reason = result.get("reason", "no reason returned")

    print(f"\nAnswer Relevancy:  {score:.2f} / 1.0  (threshold: {THRESHOLD})")
    print(f"Reason: {reason}")

    assert score >= THRESHOLD, (
        f"Relevancy score {score:.2f} is below threshold {THRESHOLD}\n"
        f"Reason: {reason}"
    )
