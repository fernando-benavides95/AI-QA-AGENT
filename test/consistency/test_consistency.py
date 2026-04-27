"""
Layer 3 — Consistency / non-determinism testing

Runs the generator multiple times with the same input and measures
variance in output (count, category distribution, score range).
Requires multiple real LLM calls. Run with: pytest -m llm
"""
import pytest

pytestmark = pytest.mark.llm
