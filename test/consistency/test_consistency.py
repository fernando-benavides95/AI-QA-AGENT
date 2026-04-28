"""
Layer 3 — Consistency / non-determinism testing

LEARNING OBJECTIVE: This layer exists to make the concept of non-determinism
tangible. In traditional QA, variance between runs is a bug. In AI testing,
variance is a property of the system — you measure it and set tolerances.

    Traditional QA:  assert count == 8
    AI testing:      assert MIN_COUNT <= count <= MAX_COUNT

HONEST CAVEAT: In a real production system, Layer 3 would not primarily live
in test code. The most meaningful consistency tests — quality score variance,
distribution drift, degradation trends — require observability over time.
You would query historical run data from LangSmith, Confident AI, or a custom
reporting layer and assert on distributions across many accumulated runs.

What lives in test code at this layer is the cheap structural invariants:
things that must never be violated on any single run, and that would signal
a prompt regression or system failure if they did. Everything else — score
distributions, trend analysis — is a dashboard and observability concern.

The thresholds below were set after an initial calibration run of 5 executions.
HOW TO RECALIBRATE: run the file, observe the printed values, update constants.

Run with: pytest -m llm
"""
import pytest
from app.graph import TestCaseGeneratorGraph
from app.tools import analyze_test_coverage

pytestmark = pytest.mark.llm

FEATURE = "Simple login page with email and password"
N_RUNS = 5

# Calibrated from 5 observed runs — test case counts: [15, 14, 14, 13, 14]
MIN_TEST_CASES = 10
MAX_TEST_CASES = 18


@pytest.fixture(scope="module")
def multiple_runs():
    """Runs the full agent loop N times. module scope: shared across all tests."""
    graph = TestCaseGeneratorGraph()
    results = []
    for i in range(N_RUNS):
        print(f"\n--- Consistency run {i + 1}/{N_RUNS} ---")
        results.append(graph.run(FEATURE))
    return results


@pytest.fixture(scope="module")
def coverage_reports(multiple_runs):
    return [
        analyze_test_coverage.invoke({"test_cases": run["test_cases"]})
        for run in multiple_runs
    ]


class TestTerminationConsistency:

    def test_all_runs_reach_approved_status(self, multiple_runs):
        for i, run in enumerate(multiple_runs):
            assert run["status"] == "approved", (
                f"Run {i + 1} did not reach approved status: {run['status']}"
            )

    def test_iteration_count_stays_within_expected_range(self, multiple_runs):
        # Calibrated baseline: always 5 iterations. Upper bound is generous to
        # allow normal variance while catching a genuine behavioural regression
        # (e.g. a prompt change that makes the critic far harder to satisfy).
        counts = [run["iterations"] for run in multiple_runs]
        print(f"\nIteration counts: {counts}  (min={min(counts)} max={max(counts)})")
        assert max(counts) <= 10, (
            f"A run took {max(counts)} iterations — above the observed baseline of 5"
        )

    def test_test_case_count_within_expected_range(self, coverage_reports):
        counts = [r["total"] for r in coverage_reports]
        print(f"\nTest case counts: {counts}")
        for i, count in enumerate(counts):
            assert MIN_TEST_CASES <= count <= MAX_TEST_CASES, (
                f"Run {i + 1} produced {count} test cases "
                f"(expected {MIN_TEST_CASES}–{MAX_TEST_CASES})"
            )
