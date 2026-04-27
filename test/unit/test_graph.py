"""
Layer 1 — Unit tests for routing logic in app/graph.py

The only project logic worth testing here is route_after_review — the function
that decides whether the agent loop continues or exits. It has three branches,
each a business rule that could have a bug:

  approved        → exit
  max iterations  → exit
  otherwise       → loop back to generate

Topology tests (does the graph have two nodes? are edges stored correctly?)
are not added — they assert that LangGraph stores what you put in, which is
testing the framework, not the code.
"""
import pytest
from langgraph.graph import END
from app.graph import route_after_review, MAX_ITERATIONS

pytestmark = pytest.mark.unit


def make_state(status: str, iterations: int) -> dict:
    return {
        "status": status,
        "iterations": iterations,
        "test_cases": [],
        "additional_considerations": [],
        "coverage_report": "",
        "feedback_history": [],
        "feature_description": "test feature",
    }


class TestRouteAfterReview:

    def test_approved_status_exits_graph(self):
        state = make_state(status="approved", iterations=1)
        assert route_after_review(state) == END

    @pytest.mark.parametrize("iterations", [1, MAX_ITERATIONS - 1])
    def test_in_progress_below_limit_loops_to_generate(self, iterations):
        # Two points in the "should loop" range:
        # - 1: first iteration (well within limit)
        # - MAX-1: last iteration before the exit boundary
        state = make_state(status="in_progress", iterations=iterations)
        assert route_after_review(state) == "generate"

    @pytest.mark.parametrize("iterations", [MAX_ITERATIONS, MAX_ITERATIONS + 1])
    def test_at_or_over_max_iterations_exits_graph(self, iterations):
        # Two exit boundary points:
        # - MAX: exactly at the limit (the defined exit point)
        # - MAX+1: over the limit — guards against a bug where iterations
        #   somehow exceeds MAX without triggering exit on the previous turn
        state = make_state(status="in_progress", iterations=iterations)
        assert route_after_review(state) == END

    def test_approved_exits_regardless_of_iteration_count(self):
        # Approval on the first iteration is valid — the critic may approve
        # immediately for simple features.
        state = make_state(status="approved", iterations=1)
        assert route_after_review(state) == END
