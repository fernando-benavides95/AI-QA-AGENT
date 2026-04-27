"""
Layer 1 — Unit tests for analyze_test_coverage (app/tools.py)

VALUE: This function sits at the deterministic boundary of the agent — it has
no randomness, no LLM involvement, and always produces the same output for the
same input. 
"""
import pytest
from app.tools import analyze_test_coverage

pytestmark = pytest.mark.unit

# The @tool decorator wraps the function in a LangChain StructuredTool object.
# We call it via .invoke() — the same way graph.py calls it — so we are testing
# the tool as it is actually used, not just the raw logic underneath.


class TestAnalyzeTestCoverageEmptyInput:
    # The LLM could theoretically return zero testcases
    # This test ensures case is handled,
    # and that missing_categories correctly reports all three as absent.

    def test_empty_list_returns_zero_total(self):
        result = analyze_test_coverage.invoke({"test_cases": []})
        assert result["total"] == 0

    def test_empty_list_reports_all_categories_missing(self):
        result = analyze_test_coverage.invoke({"test_cases": []})
        assert result["missing_categories"] == ["Edge Case", "Negative", "Positive"]

    def test_empty_list_has_no_duplicates(self):
        result = analyze_test_coverage.invoke({"test_cases": []})
        assert result["duplicate_descriptions"] == []


class TestAnalyzeTestCoverageFullSuite:
    # Happy path:  verifies that counts are accurate when all three categories are present. 
    # This is the contract the critic relies on when it reads the coverage report. 

    def test_total_count_is_accurate(self, sample_test_cases):
        result = analyze_test_coverage.invoke({"test_cases": sample_test_cases})
        assert result["total"] == len(sample_test_cases)

    def test_category_counts_are_accurate(self, sample_test_cases):
        result = analyze_test_coverage.invoke({"test_cases": sample_test_cases})
        assert result["by_category"]["Positive"] == 1
        assert result["by_category"]["Negative"] == 1
        assert result["by_category"]["Edge Case"] == 1

    def test_no_missing_categories_when_all_present(self, sample_test_cases):
        result = analyze_test_coverage.invoke({"test_cases": sample_test_cases})
        assert result["missing_categories"] == []


class TestAnalyzeTestCoverageMissingCategory:
    # The critic uses missing_categories to decide whether to approve.
    # This test confirms that if the generator skips edgecases, the critic will be told so

    def test_detects_missing_edge_case_category(self):
        test_cases = [
            {"id": "TC01", "description": "Valid login", "input_data": "...",
             "expected_output": "...", "priority": "High", "category": "Positive"},
            {"id": "TC02", "description": "Invalid login", "input_data": "...",
             "expected_output": "...", "priority": "High", "category": "Negative"},
        ]
        result = analyze_test_coverage.invoke({"test_cases": test_cases})
        assert "Edge Case" in result["missing_categories"]

    def test_missing_categories_list_is_sorted(self):
        test_cases = [
            {"id": "TC01", "description": "Valid login", "input_data": "...",
             "expected_output": "...", "priority": "High", "category": "Positive"},
        ]
        result = analyze_test_coverage.invoke({"test_cases": test_cases})
        assert result["missing_categories"] == sorted(result["missing_categories"])
