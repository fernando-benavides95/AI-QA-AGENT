"""
Layer 1 — Deterministic unit tests for app/tools.py

Tests analyze_test_coverage with no LLM calls.
The @tool decorator exposes the underlying function via .invoke().
"""
import pytest
from app.tools import analyze_test_coverage

pytestmark = pytest.mark.unit
