"""
Layer 1 — Deterministic unit tests for app/models.py

Tests that Pydantic schemas enforce their contracts.
No LLM calls — pure schema validation.
"""
import pytest
from app.models import TestCase, TestSuite, CriticResponse

pytestmark = pytest.mark.unit
