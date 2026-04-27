import pytest
from app.agents import AgentState


@pytest.fixture
def sample_feature_description() -> str:
    return "Simple login page with email and password"


@pytest.fixture
def sample_test_cases() -> list:
    return [
        {
            "id": "TC01",
            "description": "Successful login with valid credentials",
            "input_data": "email: user@example.com, password: ValidPass123",
            "expected_output": "Redirect to dashboard",
            "priority": "High",
            "category": "Positive",
        },
        {
            "id": "TC02",
            "description": "Login failure with invalid password",
            "input_data": "email: user@example.com, password: WrongPass",
            "expected_output": "Error: Invalid credentials",
            "priority": "High",
            "category": "Negative",
        },
        {
            "id": "TC03",
            "description": "Login with empty fields",
            "input_data": "email: '', password: ''",
            "expected_output": "Validation error displayed",
            "priority": "Medium",
            "category": "Edge Case",
        },
    ]


@pytest.fixture
def initial_state(sample_feature_description) -> AgentState:
    return {
        "test_cases": [],
        "additional_considerations": [],
        "coverage_report": "",
        "feedback_history": [],
        "iterations": 0,
        "feature_description": sample_feature_description,
        "status": "in_progress",
    }
