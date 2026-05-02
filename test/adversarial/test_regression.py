"""
Layer 4 — Prompt regression testing

CONCEPT: When you change a prompt, you change behaviour. Without a regression
suite you have no way to know if the change improved things, degraded them, or
both — improving one scenario while breaking another.

This file is the safety net for prompt changes. Run it before and after any
modification to the generator or critic prompts. If scores drop below the
golden dataset thresholds, the change is a regression.

HOW THE GOLDEN DATASET WORKS:
  fixtures/golden_dataset.json contains a small set of representative feature
  descriptions with a minimum acceptable quality score for each. These represent
  your baseline — what "good enough" looks like for the current system.

  When to update the dataset:
    - Add a new entry when you onboard a new feature category the system must handle
    - Raise a threshold when you make a targeted improvement and verify the gain
    - Never lower a threshold to make a failing test pass — that's defeating the point

WORKFLOW FOR SAFE PROMPT CHANGES:
  1. Run this file: pytest test/adversarial/test_regression.py -m llm
  2. Record the scores (current baseline)
  3. Make your prompt change
  4. Run again — any score below its threshold is a regression
  5. If all scores hold or improve, the change is safe to ship

Run with: pytest -m llm
"""
import json
import re
import pytest
from pathlib import Path
from app.graph import TestCaseGeneratorGraph
from app.agents import CriticAgent

pytestmark = pytest.mark.llm

GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "fixtures" / "golden_dataset.json"

RELEVANCY_PROMPT = """
You are an expert software testing evaluator.

Feature: {feature}
Generated test cases: {output}

Score how relevant the test cases are to the feature (0.0 to 1.0).
Respond with ONLY valid JSON:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def graph():
    return TestCaseGeneratorGraph()


class TestPromptRegression:

    @pytest.mark.parametrize(
        "entry",
        load_golden_dataset(),
        ids=[e["id"] for e in load_golden_dataset()],
    )
    def test_quality_meets_golden_threshold(self, entry, judge, graph):
        """
        Runs the agent against each golden dataset entry and asserts the output
        meets the defined minimum quality score.

        A failure here means a prompt change has degraded quality for this
        feature category — the change should be reviewed before shipping.
        """
        feature = entry["feature"]
        threshold = entry["min_relevancy_score"]

        result = graph.run(feature)
        output = CriticAgent()._format_for_review(result["test_cases"])

        evaluation = judge.evaluate(
            RELEVANCY_PROMPT.format(feature=feature, output=output)
        )
        score = evaluation["score"]
        reason = evaluation.get("reason", "")

        print(f"\n[{entry['id']}] {entry['description']}")
        print(f"  Score: {score:.2f}  Threshold: {threshold}  {'PASS' if score >= threshold else 'FAIL'}")
        print(f"  Reason: {reason}")

        assert score >= threshold, (
            f"[{entry['id']}] Regression detected: score {score:.2f} "
            f"is below threshold {threshold}\n"
            f"Feature: {feature}\n"
            f"Reason: {reason}"
        )
