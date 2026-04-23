from dotenv import load_dotenv
from app.graph import TestCaseGeneratorGraph

load_dotenv()

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
            print("Generated Test Cases:")
            print(final_state["test_cases"])
            print("-" * 40)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your API key, model configuration, and network connection.")
            print("-" * 40)

if __name__ == "__main__":
    run_test_case_generator()