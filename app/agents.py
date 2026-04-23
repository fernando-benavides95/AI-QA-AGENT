from typing import List, Dict, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")

# --- Agent State ---
class AgentState(TypedDict):
    test_cases: str
    feedback: str
    iterations: int
    feature_description: str
    status: str  # "in_progress" | "approved" | "max_iterations_reached"

# --- Generator Agent ---
class GeneratorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert test case generator. Create comprehensive test cases for a given software feature, covering positive, negative, and edge cases. For each test case provide: a description, input data, expected output, priority (High/Medium/Low), and category (Positive/Negative/Edge Case). Format your output as a clean markdown table."),
            ("human", "Generate test cases for the following feature: {feature_description}\n\nPrevious feedback: {feedback}"),
        ])

    def generate_test_cases(self, state: AgentState) -> AgentState:
        chain = self.prompt | self.llm
        llm_response = chain.invoke({
            "feature_description": state["feature_description"],
            "feedback": state["feedback"]
        })

        content = llm_response.content
        if isinstance(content, list):
            content = content[0]["text"]

        return {
            **state,
            "test_cases": content,
            "iterations": state.get("iterations", 0) + 1,
            "status": "in_progress"
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
                "You are an expert test case critic. Review the provided test cases "
                "for coverage, boundary conditions, and completness"
                "and judge their quality relative to the complexity of the feature being tested — "
                "If the test cases adequately cover the core positive, negative, and edge cases "
                "respond with APPROVED followed by one brief line "
                "listing the coverage categories you verified (e.g. 'APPROVED — verified: happy path, "
                "auth failures, input validation, security, edge cases'). "
                "Otherwise, give concise, actionable improvements. "
                "Do not rewrite the test cases yourself. "
                "On iteration {iteration} of a maximum of 10: if you are on iteration 5 or higher, "
                "only withhold approval for a genuine critical gap, not minor improvements."
            )),
            ("human", (
                "Feature being tested:\n{feature_description}\n\n"
                "Test cases to review:\n{test_cases}"
            )),
        ])

    def review_test_cases(self, state: AgentState) -> AgentState:
        chain = self.prompt | self.llm
        llm_response = chain.invoke({
            "test_cases": state["test_cases"],
            "feature_description": state["feature_description"],
            "iteration": state.get("iterations", 1),
        })

        content = llm_response.content
        if isinstance(content, list):
            content = content[0]["text"]

        if "APPROVED" in content.upper():
            return {**state, "feedback": "APPROVED", "status": "approved"}
        else:
            return {**state, "feedback": content, "status": "in_progress"}