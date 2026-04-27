"""
Shared fixtures for Layer 2 evaluation tests.

GeminiJudge wraps the project's existing LLM as DeepEval's evaluation model.
The agent_output fixture runs the full graph ONCE per session so all evaluation
tests share the same output — avoiding repeated API calls and cost.
"""
import os
import pytest
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM
from app.graph import TestCaseGeneratorGraph
from app.agents import CriticAgent

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Feature used as the input for all Layer 2 evaluations.
# Deliberately mid-complexity: simple enough to have a clear expected output,
# rich enough to have meaningful positive/negative/edge coverage requirements.
EVALUATION_FEATURE = "Simple login page with email and password"


class GeminiJudge(DeepEvalBaseLLM):
    """
    Wraps Gemini as DeepEval's evaluation judge.

    NOTE (best practice): using the same model family to generate AND judge
    introduces self-evaluation bias — the model tends to score its own output
    higher than an independent judge would. In production, use a different
    provider as the judge (Anthropic, OpenAI, etc.).
    """

    def __init__(self):
        self._model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,  # deterministic for evaluation
        )

    def load_model(self):
        return self._model

    def _extract_text(self, content) -> str:
        if isinstance(content, list):
            return content[0].get("text", "") if content else ""
        return content

    def generate(self, prompt: str) -> str:
        return self._extract_text(self._model.invoke(prompt).content)

    async def a_generate(self, prompt: str) -> str:
        response = await self._model.ainvoke(prompt)
        return self._extract_text(response.content)

    def get_model_name(self) -> str:
        return GEMINI_MODEL_NAME


@pytest.fixture(scope="session")
def gemini_judge():
    return GeminiJudge()


@pytest.fixture(scope="session")
def agent_output():
    """
    Runs the full generator-critic loop once and caches the result.
    All evaluation tests in this session share this output.
    """
    graph = TestCaseGeneratorGraph()
    return graph.run(EVALUATION_FEATURE)


@pytest.fixture(scope="session")
def formatted_output(agent_output):
    """
    Formats the structured test cases into readable text for DeepEval.
    Uses the same format the critic sees — descriptive enough for the judge
    to evaluate relevancy and coverage quality.
    """
    critic = CriticAgent()
    return critic._format_for_review(agent_output["test_cases"])
