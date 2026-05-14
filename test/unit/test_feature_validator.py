"""
Layer 1 — Unit tests for app/feature_validator.py

Five groups, same pattern as test_agents.py:

1. Routing functions — pure, module-level, no LLM calls
2. _extract_text_content — handles Gemini's content block format
3. _extract_refined_description — locates the finalised description in message history
4. _build_system_prompt — turn awareness and hijacking guard
5. _validate_node state transitions — validator LLM mocked, state logic isolated
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from app.feature_validator import (
    FeatureValidatorGraph,
    ValidatorState,
    _route_after_agent,
    _route_after_tools,
    _route_after_human_approval,
    _extract_text_content,
    _extract_refined_description,
    _build_system_prompt,
    _APPROVAL_KEYWORDS,
    MAX_CLARIFICATION_TURNS,
    MAX_VALIDATION_ATTEMPTS,
)
from app.models import FeatureValidation

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> dict:
    base = {
        "messages": [],
        "validation_attempts": 0,
        "is_valid": False,
        "conversation_turns": 0,
        "refined_description": "",
    }
    base.update(overrides)
    return base


def _ai_with_tool(tool_name: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "id": "tc-1", "args": {"query": "test"}}],
    )


# ---------------------------------------------------------------------------
# Group 1 — Routing functions
# ---------------------------------------------------------------------------

class TestRouteAfterAgent:
    """
    After the agent responds, we branch on whether it called a tool or returned text.
    A tool call (any tool) goes to the tools node for execution.
    A plain text response goes to human_input to interrupt and wait for the user.
    """

    def test_tool_call_routes_to_tools(self):
        state = _state(messages=[_ai_with_tool("search_testing_patterns")])
        assert _route_after_agent(state) == "tools"

    def test_finalize_tool_call_also_routes_to_tools(self):
        state = _state(messages=[_ai_with_tool("finalize_feature")])
        assert _route_after_agent(state) == "tools"

    def test_text_response_routes_to_human_input(self):
        state = _state(messages=[AIMessage(content="What system is this for?")])
        assert _route_after_agent(state) == "human_input"

    def test_empty_tool_calls_routes_to_human_input(self):
        state = _state(messages=[AIMessage(content="Tell me more.", tool_calls=[])])
        assert _route_after_agent(state) == "human_input"


class TestRouteAfterTools:
    """
    After a tool executes, we decide whether the conversation is done or continues.
    finalize_feature signals the agent is ready — route to the approval gate.
    Any other tool (e.g. web search) means the agent is still gathering context — route back to agent.
    The message traversal test matters: we read the most recent AIMessage with tool calls,
    not just any message in history.
    """

    def test_finalize_feature_routes_to_human_approval(self):
        state = _state(messages=[_ai_with_tool("finalize_feature")])
        assert _route_after_tools(state) == "human_approval"

    def test_search_tool_routes_to_agent(self):
        state = _state(messages=[_ai_with_tool("search_testing_patterns")])
        assert _route_after_tools(state) == "agent"

    def test_uses_most_recent_ai_message_with_tool_calls(self):
        # Earlier message had search, latest has finalize — should route to approval
        search_msg = _ai_with_tool("search_testing_patterns")
        finalize_msg = _ai_with_tool("finalize_feature")
        state = _state(messages=[search_msg, finalize_msg])
        assert _route_after_tools(state) == "human_approval"


class TestRouteAfterHumanApproval:
    """
    The approval gate asks the user to confirm the refined description before handing off.
    Any keyword in _APPROVAL_KEYWORDS ends the validator session and triggers the generator.
    Anything else — a correction, extra context, or empty input — routes back to the agent.
    The parametrize tests each keyword individually so a failure pinpoints exactly which word broke.
    Case-insensitive and whitespace handling are tested because that's what real users type.
    """

    @pytest.mark.parametrize("keyword", sorted(_APPROVAL_KEYWORDS))
    def test_approval_keyword_routes_to_end(self, keyword):
        state = _state(messages=[HumanMessage(content=keyword)])
        assert _route_after_human_approval(state) == END

    def test_approval_keyword_case_insensitive(self):
        state = _state(messages=[HumanMessage(content="YES")])
        assert _route_after_human_approval(state) == END

    def test_approval_keyword_with_whitespace(self):
        state = _state(messages=[HumanMessage(content="  yes  ")])
        assert _route_after_human_approval(state) == END

    def test_correction_text_routes_to_agent(self):
        state = _state(messages=[HumanMessage(content="Add that it must support SSO as well")])
        assert _route_after_human_approval(state) == "agent"

    def test_empty_approval_message_routes_to_agent(self):
        state = _state(messages=[HumanMessage(content="")])
        assert _route_after_human_approval(state) == "agent"


# ---------------------------------------------------------------------------
# Group 2 — _extract_text_content
# ---------------------------------------------------------------------------

class TestExtractTextContent:
    """
    Gemini returns AIMessage.content as a list of typed content blocks rather than a plain string
    when the model uses extended reasoning. This helper normalises that to a string.
    It is a compatibility shim for this model's behaviour — not core domain logic —
    but it owns a real responsibility: if it breaks, the UI prints raw dicts to the user.
    """

    def test_plain_string_returned_as_is(self):
        assert _extract_text_content("hello") == "hello"

    def test_gemini_content_block_list_extracts_text(self):
        content = [{"type": "text", "text": "What system is this for?", "extras": {"signature": "abc"}}]
        assert _extract_text_content(content) == "What system is this for?"

    def test_multiple_text_blocks_joined(self):
        content = [
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ]
        result = _extract_text_content(content)
        assert "First part." in result
        assert "Second part." in result

    def test_non_text_blocks_excluded(self):
        content = [
            {"type": "thinking", "text": "internal reasoning"},
            {"type": "text", "text": "visible response"},
        ]
        result = _extract_text_content(content)
        assert "visible response" in result
        assert "internal reasoning" not in result

    def test_empty_list_returns_empty_string(self):
        assert _extract_text_content([]) == ""


# ---------------------------------------------------------------------------
# Group 3 — _extract_refined_description
# ---------------------------------------------------------------------------

class TestExtractRefinedDescription:
    """
    The refined description is what gets passed to the Generator — the entire handoff depends on it.
    The function reads the message history to find it, in priority order:
    1. A ToolMessage from finalize_feature (the happy path — agent explicitly finalised)
    2. The last HumanMessage (fallback — session ended without finalising)
    3. Empty string (nothing usable in history)
    ToolMessages from other tools (e.g. search results) must be ignored.
    """

    def test_extracts_from_finalize_feature_tool_message(self):
        tool_msg = ToolMessage(
            content="A login page with email and password.",
            tool_call_id="tc-1",
            name="finalize_feature",
        )
        state = {"messages": [HumanMessage(content="login page"), tool_msg]}
        assert _extract_refined_description(state) == "A login page with email and password."

    def test_falls_back_to_last_human_message_when_no_tool_message(self):
        state = {"messages": [HumanMessage(content="a shopping cart feature")]}
        assert _extract_refined_description(state) == "a shopping cart feature"

    def test_ignores_tool_messages_from_other_tools(self):
        other_msg = ToolMessage(
            content="Some search results.",
            tool_call_id="tc-1",
            name="search_testing_patterns",
        )
        human_msg = HumanMessage(content="file upload feature")
        state = {"messages": [human_msg, other_msg]}
        assert _extract_refined_description(state) == "file upload feature"

    def test_prefers_finalize_tool_message_over_human_message(self):
        human_msg = HumanMessage(content="original vague description")
        tool_msg = ToolMessage(
            content="Refined: a file upload feature with 5MB limit.",
            tool_call_id="tc-1",
            name="finalize_feature",
        )
        state = {"messages": [human_msg, tool_msg]}
        assert _extract_refined_description(state) == "Refined: a file upload feature with 5MB limit."

    def test_returns_empty_string_when_no_messages(self):
        assert _extract_refined_description({"messages": []}) == ""


# ---------------------------------------------------------------------------
# Group 4 — _build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """
    The system prompt is rebuilt on every agent turn — it tells the agent which turn it's on
    and changes behaviour at the final turn (instructs it to stop asking and call finalize_feature).
    It also carries the hijacking guard that tells the model to treat user input as data, not instructions.
    Same pattern as the Critic's iteration-awareness — the prompt itself enforces the loop exit condition.
    """

    def test_mid_turn_includes_current_turn_number(self):
        prompt = _build_system_prompt(current_turn=1)
        assert "turn 1" in prompt
        assert str(MAX_CLARIFICATION_TURNS) in prompt

    def test_mid_turn_does_not_say_final(self):
        prompt = _build_system_prompt(current_turn=1)
        assert "final clarification turn" not in prompt

    def test_final_turn_instructs_to_synthesize_and_finalize(self):
        prompt = _build_system_prompt(current_turn=MAX_CLARIFICATION_TURNS)
        assert "final clarification turn" in prompt
        assert "finalize_feature" in prompt

    def test_prompt_contains_hijacking_guard(self):
        prompt = _build_system_prompt(current_turn=1)
        assert "role is fixed" in prompt
        assert "Ignore" in prompt


# ---------------------------------------------------------------------------
# Group 5 — _validate_node state transitions (mocked validator LLM)
# ---------------------------------------------------------------------------

class TestValidateNodeTransitions:
    """
    The validate node runs a separate LLM call (structured output, FeatureValidation) before the
    conversation starts. It acts as a classifier: is this actually a software feature?
    The LLM is mocked so these tests are deterministic — we isolate the state transition logic
    from the model entirely, same pattern used in TestGeneratorStateTransformation and
    TestCriticStateTransformation in test_agents.py.
    Key behaviours: valid input sets is_valid and adds nothing to messages;
    invalid input increments attempts and adds an AIMessage; at max attempts the message includes an example.
    """

    @pytest.fixture
    def graph(self):
        return FeatureValidatorGraph()

    def _make_state(self, content: str, attempts: int = 0) -> dict:
        return {
            "messages": [HumanMessage(content=content)],
            "validation_attempts": attempts,
            "is_valid": False,
            "conversation_turns": 0,
            "refined_description": "",
        }

    def _mock_validator(self, graph, is_feature: bool, feedback: str = "Some feedback."):
        graph._validator = MagicMock()
        graph._validator.invoke.return_value = FeatureValidation(
            is_feature=is_feature, feedback=feedback
        )

    def test_valid_feature_sets_is_valid_true(self, graph):
        self._mock_validator(graph, is_feature=True)
        result = graph._validate_node(self._make_state("login page"))
        assert result["is_valid"] is True

    def test_valid_feature_does_not_add_ai_message(self, graph):
        self._mock_validator(graph, is_feature=True)
        result = graph._validate_node(self._make_state("login page"))
        assert "messages" not in result

    def test_invalid_feature_sets_is_valid_false(self, graph):
        self._mock_validator(graph, is_feature=False, feedback="This is not a software feature.")
        result = graph._validate_node(self._make_state("a chocolate cake recipe"))
        assert result["is_valid"] is False

    def test_invalid_feature_increments_attempts(self, graph):
        self._mock_validator(graph, is_feature=False, feedback="Not a software feature.")
        result = graph._validate_node(self._make_state("not a feature", attempts=0))
        assert result["validation_attempts"] == 1

    def test_invalid_first_attempt_adds_reprompt_message(self, graph):
        self._mock_validator(graph, is_feature=False, feedback="Not a software feature.")
        result = graph._validate_node(self._make_state("not a feature", attempts=0))
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_invalid_at_max_attempts_includes_example(self, graph):
        self._mock_validator(graph, is_feature=False, feedback="Still not a software feature.")
        result = graph._validate_node(
            self._make_state("still not a feature", attempts=MAX_VALIDATION_ATTEMPTS - 1)
        )
        assert "example" in result["messages"][0].content.lower()

    def test_invalid_first_attempt_does_not_include_example(self, graph):
        self._mock_validator(graph, is_feature=False, feedback="Not a software feature.")
        result = graph._validate_node(self._make_state("not a feature", attempts=0))
        assert "example" not in result["messages"][0].content.lower()
