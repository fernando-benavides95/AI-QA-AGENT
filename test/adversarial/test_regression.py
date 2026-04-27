"""
Layer 4 — Prompt regression testing

Runs the agent against the golden dataset (fixtures/golden_dataset.json)
and asserts output quality meets defined thresholds. Run after any
prompt change to catch regressions.
Requires real LLM calls. Run with: pytest -m llm
"""
import pytest

pytestmark = pytest.mark.llm
