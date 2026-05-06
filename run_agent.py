from dotenv import load_dotenv
from app.graph import TestCaseGeneratorGraph
from app.feature_validator import FeatureValidatorGraph

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


def run():
    print("=== AI QA Agent ===")
    print("Describe the feature you want to test. Type 'exit' to quit.")
    print("-" * 40)

    validator = FeatureValidatorGraph()
    generator = TestCaseGeneratorGraph()

    while True:
        initial_input = input("\nYou: ").strip()

        if initial_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        if not initial_input:
            continue

        try:
            print("\n[Feature Validator] Let me ask a few questions to refine the feature...\n")
            refined_description = validator.run(initial_input)

            if not refined_description:
                continue

            print("\n[Generator] Generating your test suite...\n")
            final_state = generator.run(refined_description)

            print("\n--- Generation Complete ---")
            print(f"Status: {final_state['status']}")
            print(f"Completed in {final_state['iterations']} iteration(s)\n")
            print(_render_test_suite(final_state))
            print("-" * 40)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your API keys and network connection.")
            print("-" * 40)


if __name__ == "__main__":
    run()
