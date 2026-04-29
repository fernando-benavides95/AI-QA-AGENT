"""Shared fixtures for Layer 4 adversarial and regression tests."""
import os
import re
import json
import pytest
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.graph import TestCaseGeneratorGraph

load_dotenv()


class LLMJudge:
    """Same judge pattern as Layer 2 — reused here to score adversarial outputs."""

    def __init__(self):
        self._model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

    def _extract_text(self, content) -> str:
        if isinstance(content, list):
            return content[0].get("text", "") if content else ""
        return content

    def evaluate(self, prompt: str) -> dict:
        raw = self._extract_text(self._model.invoke(prompt).content)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned non-JSON response:\n{raw}")


@pytest.fixture(scope="session")
def graph():
    return TestCaseGeneratorGraph()


@pytest.fixture(scope="session")
def judge():
    return LLMJudge()
