from collections import Counter
from langchain_core.tools import tool


@tool
def analyze_test_coverage(test_cases: list) -> dict:
    """
    Analyze a list of test case dicts and return objective coverage statistics.

    Checks category distribution (Positive / Negative / Edge Case), priority
    distribution (High / Medium / Low), total count, and duplicate descriptions.
    Use this before reviewing a test suite to ground qualitative judgements in
    concrete data.
    """
    if not test_cases:
        return {
            "total": 0,
            "by_category": {},
            "by_priority": {},
            "missing_categories": ["Edge Case", "Negative", "Positive"],
            "duplicate_descriptions": [],
        }

    categories = Counter(tc.get("category", "") for tc in test_cases)
    priorities = Counter(tc.get("priority", "") for tc in test_cases)

    descriptions = [tc.get("description", "").lower().strip() for tc in test_cases]
    duplicates = [d for d, n in Counter(descriptions).items() if n > 1 and d]

    expected = {"Positive", "Negative", "Edge Case"}
    missing = sorted(expected - set(categories.keys()))

    return {
        "total": len(test_cases),
        "by_category": dict(categories),
        "by_priority": dict(priorities),
        "missing_categories": missing,
        "duplicate_descriptions": duplicates,
    }


def format_coverage_report(analysis: dict) -> str:
    """Render the analyze_test_coverage output as a compact human-readable summary."""
    lines = [
        f"Total test cases: {analysis['total']}",
        f"By category: {analysis['by_category']}",
        f"By priority: {analysis['by_priority']}",
    ]
    if analysis["missing_categories"]:
        lines.append(f"Missing categories: {', '.join(analysis['missing_categories'])}")
    if analysis["duplicate_descriptions"]:
        lines.append(f"Duplicate descriptions detected: {len(analysis['duplicate_descriptions'])}")
    else:
        lines.append("No duplicate descriptions.")
    return "\n".join(lines)
