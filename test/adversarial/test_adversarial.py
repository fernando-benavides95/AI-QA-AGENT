"""
Layer 4 — Adversarial input testing

Feeds edge-case and malicious feature descriptions to expose
failure modes: ambiguous inputs, contradictory requirements,
prompt injection attempts.
Requires real LLM calls. Run with: pytest -m llm
"""
import pytest

pytestmark = pytest.mark.llm
