"""
Layer 2 — LLM evaluation: Answer Relevancy (DeepEval)

Asserts that generated test cases are relevant to the feature description.
Requires a real LLM call. Run with: pytest -m llm
"""
import pytest

pytestmark = pytest.mark.llm
