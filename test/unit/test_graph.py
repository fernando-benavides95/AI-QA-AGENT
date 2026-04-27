"""
Layer 1 — Deterministic unit tests for app/graph.py

Tests the compiled LangGraph topology: nodes, edges, entry point.
Compiling the graph does not make LLM calls.
"""
import pytest
from app.graph import TestCaseGeneratorGraph

pytestmark = pytest.mark.unit
