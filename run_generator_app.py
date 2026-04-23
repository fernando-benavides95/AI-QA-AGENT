from dotenv import load_dotenv
from app.graph import TestCaseGeneratorGraph

load_dotenv()


def _render_test_suite(state: dict) -> str:
    col = {"id": 6, "description": 36, "input_data": 28, "expected_output": 32, "priority": 8, "category": 12}

    def row(vals: dict) -> str:
        return "| " + " | ".join(str(vals.get(k, ""))[:w].ljust(w) for k, w in col.items()) + " |"

    header = row({k: k.replace("_", " ").title() for k in col})
    separator = "| " + " | ".join("-" * w for w in col.values()) + " |"
    lines = [header, separator] + [row(tc) for tc in state.get("test_cases", [])]

    if state.get("additional_considerations"):
        lines += ["\nAdditional Considerations:"]
        lines += [f"  - {c}" for c in state["additional_considerations"]]

    if state.get("status") == "approved":
        lines += ["\nAPPROVED"]

    return "\n".join(lines)


def run_test_case_generator():
    print("--- Test Case Generator ---")
    print("Enter the feature description for which you want to generate test cases.")
    print("Type 'exit' or 'quit' to stop the application.")
    print("-" * 40)

    graph = TestCaseGeneratorGraph()

    while True:
        feature_description = input("\nEnter feature description (or 'exit' to quit): ").strip()

        if feature_description.lower() in ["exit", "quit"]:
            print("Exiting application.")
            break

        if not feature_description:
            print("Feature description cannot be empty. Please try again.")
            continue

        try:
            final_state = graph.run(feature_description)

            print("\n--- Generation Complete ---")
            print(f"Status: {final_state['status']}")
            print(f"Completed in {final_state['iterations']} iteration(s)\n")
            print(_render_test_suite(final_state))
            print("-" * 40)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your API key, model configuration, and network connection.")
            print("-" * 40)

if __name__ == "__main__":
    run_test_case_generator()