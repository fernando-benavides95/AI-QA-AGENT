"""
Shared fixtures for Layer 2 evaluation tests.

LLMJudge wraps Gemini directly — no evaluation framework needed.
The judge receives a prompt, returns a JSON score, and we assert on that score.
This makes the full evaluation pipeline transparent and readable.

The agent_output fixture runs the full graph ONCE per session so all evaluation
tests share the same output — avoiding repeated API calls and cost.

TBD: REPORTING GAP: results are currently console-only (printed during the run).
Alternatives to revisit when a reporting layer is needed:
  - LangSmith (integrates natively with LangChain, accepts personal accounts)
  - Self-hosted: capture judge responses and write timestamped JSON to reports/

NOTE (best practice): using the same model family to generate AND judge
introduces self-evaluation bias — the model tends to approve its own output.
In production use a different provider as the judge (Anthropic, OpenAI, etc.).
"""
import os
import re
import json
import pytest
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.graph import TestCaseGeneratorGraph
from app.agents import CriticAgent

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Feature used as input for all Layer 2 evaluations.
# Mid-complexity: clear enough to have expected coverage, rich enough to
# require positive/negative/edge cases.
EVALUATION_FEATURE = "Simple login page with email and password"


class LLMJudge:
    """
    Manual LLM-as-judge evaluator.

    Sends a scoring prompt to Gemini and parses the JSON response.
    Each evaluation prompt defines its own scoring criteria — the judge
    scores against whatever rubric the prompt describes.
    """

    def __init__(self):
        self._model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

    def _extract_text(self, content) -> str:
        if isinstance(content, list):
            return content[0].get("text", "") if content else ""
        return content

    def evaluate(self, prompt: str) -> dict:
        """
        Sends prompt to the judge and returns parsed result.
        Expected return shape: {"score": float, "reason": str, ...}
        Raises ValueError if the response is not valid JSON.
        """
        raw = self._extract_text(self._model.invoke(prompt).content)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned non-JSON response:\n{raw}")


@pytest.fixture(scope="session")
def judge():
    return LLMJudge()


@pytest.fixture(scope="session")
def agent_output():
    """Runs the full generator-critic loop once and caches the result."""
    graph = TestCaseGeneratorGraph()
    return graph.run(EVALUATION_FEATURE)


@pytest.fixture(scope="session")
def formatted_output(agent_output):
    """Formats structured test cases as readable text for the judge prompt."""
    critic = CriticAgent()
    return critic._format_for_review(agent_output["test_cases"])
