"""
Layer 1 — Unit tests for app/agents.py

Tests AgentState shape after node execution using a mocked LLM.
The mock returns a fixed valid response so no API call is made.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.agents import GeneratorAgent, CriticAgent

pytestmark = pytest.mark.unit
