"""
Layer 2 — LLM evaluation: G-Eval with custom rubric (DeepEval)

Scores test suite quality against a rubric that defines what good
coverage means (positive, negative, edge case, security presence).
Requires a real LLM call. Run with: pytest -m llm
"""
import pytest

pytestmark = pytest.mark.llm
