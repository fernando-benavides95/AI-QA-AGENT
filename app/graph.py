from app.agents import GeneratorAgent, CriticAgent, AgentState

MAX_ITERATIONS = 10

class TestCaseGeneratorGraph:
    def __init__(self):
        self.generator = GeneratorAgent()
        self.critic = CriticAgent()

    def run(self, feature_description: str) -> AgentState:
        state: AgentState = {
            "test_cases": "",
            "feedback": "No previous feedback.",
            "iterations": 0,
            "feature_description": feature_description,
            "status": "in_progress"
        }

        while state["iterations"] < MAX_ITERATIONS:
            print(f"\n[Iteration {state['iterations'] + 1}] Generator is working...")
            state = self.generator.generate_test_cases(state)

            print(f"[Iteration {state['iterations']}] Critic is reviewing...")
            state = self.critic.review_test_cases(state)

            if state["status"] == "approved":
                print(f"[Iteration {state['iterations']}] Critic approved the test cases!")
                break
            else:
                print(f"[Iteration {state['iterations']}] Critic requested revisions. Feeding back to generator...")

        if state["status"] != "approved":
            print(f"\n[Max iterations ({MAX_ITERATIONS}) reached. Returning best result so far.]")
            state = {**state, "status": "max_iterations_reached"}

        return state