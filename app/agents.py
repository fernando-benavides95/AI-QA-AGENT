from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from app.models import TestSuite, CriticResponse

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")


# --- Agent State ---
class AgentState(TypedDict):
    test_cases: list                  # List[dict] — serialised TestCase objects
    additional_considerations: list   # List[str]
    feedback: str
    iterations: int
    feature_description: str
    status: str                       # "in_progress" | "approved" | "max_iterations_reached"


# --- Generator Agent ---
class GeneratorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert test case generator. "
                "Create comprehensive test cases for a given software feature, "
                "covering positive, negative, and edge cases. "
                "Include security and boundary considerations where relevant. "
                "Return structured output only — do not add prose outside the schema."
            )),
            ("human", (
                "Feature: {feature_description}\n\n"
                "Previous critic feedback (address all points): {feedback}"
            )),
        ])

    def generate_test_cases(self, state: AgentState) -> AgentState:
        chain = self.prompt | self.llm.with_structured_output(TestSuite)
        result: TestSuite = chain.invoke({
            "feature_description": state["feature_description"],
            "feedback": state["feedback"],
        })

        return {
            **state,
            "test_cases": [tc.model_dump() for tc in result.test_cases],
            "additional_considerations": result.additional_considerations,
            "iterations": state.get("iterations", 0) + 1,
            "status": "in_progress",
        }


# --- Critic Agent ---
class CriticAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert test case critic. "
                "Judge the test suite's quality relative to the complexity of the feature — "
                "simple features need fewer cases than complex ones. "
                "Approve if the suite adequately covers core positive, negative, and edge cases. "
                "If not approved, give at most 3 concise actionable improvements. "
                "On iteration {iteration} of a maximum of 10: from iteration 5 onward, "
                "only withhold approval for a genuine critical gap."
            )),
            ("human", (
                "Feature: {feature_description}\n\n"
                "Test cases to review:\n{test_cases}"
            )),
        ])

    def _format_for_review(self, test_cases: list) -> str:
        lines = []
        for tc in test_cases:
            lines.append(
                f"[{tc['id']}] ({tc['priority']}, {tc['category']}) "
                f"{tc['description']} | Input: {tc['input_data']} | Expected: {tc['expected_output']}"
            )
        return "\n".join(lines)

    def review_test_cases(self, state: AgentState) -> AgentState:
        chain = self.prompt | self.llm.with_structured_output(CriticResponse)
        result: CriticResponse = chain.invoke({
            "test_cases": self._format_for_review(state["test_cases"]),
            "feature_description": state["feature_description"],
            "iteration": state.get("iterations", 1),
        })

        if result.gaps_identified:
            print(f"  Gaps found: {'; '.join(result.gaps_identified)}")

        if result.approved:
            summary = f"APPROVED — verified: {', '.join(result.verified_categories)}"
            return {**state, "feedback": summary, "status": "approved"}
        else:
            return {**state, "feedback": result.feedback, "status": "in_progress"}
