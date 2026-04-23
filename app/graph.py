from langgraph.graph import StateGraph, END

from app.agents import GeneratorAgent, CriticAgent, AgentState

MAX_ITERATIONS = 10


class TestCaseGeneratorGraph:
    def __init__(self):
        generator = GeneratorAgent()
        critic = CriticAgent()

        def generate_node(state: AgentState) -> AgentState:
            print(f"\n[Iteration {state['iterations'] + 1}] Generator is working...")
            return generator.generate_test_cases(state)

        def review_node(state: AgentState) -> AgentState:
            print(f"[Iteration {state['iterations']}] Critic is reviewing...")
            result = critic.review_test_cases(state)
            if result["status"] == "approved":
                print(f"[Iteration {state['iterations']}] Critic approved the test cases!")
            else:
                print(f"[Iteration {state['iterations']}] Critic requested revisions. Feeding back to generator...")
            return result

        def route_after_review(state: AgentState) -> str:
            if state["status"] == "approved" or state["iterations"] >= MAX_ITERATIONS:
                return END
            return "generate"

        builder = StateGraph(AgentState)
        builder.add_node("generate", generate_node)
        builder.add_node("review", review_node)
        builder.set_entry_point("generate")
        builder.add_edge("generate", "review")
        builder.add_conditional_edges("review", route_after_review)

        self.graph = builder.compile()

    def run(self, feature_description: str) -> AgentState:
        initial_state: AgentState = {
            "test_cases": [],
            "additional_considerations": [],
            "feedback": "No previous feedback.",
            "iterations": 0,
            "feature_description": feature_description,
            "status": "in_progress",
        }

        final_state = self.graph.invoke(initial_state)

        if final_state["status"] != "approved":
            print(f"\n[Max iterations ({MAX_ITERATIONS}) reached. Returning best result so far.]")
            final_state = {**final_state, "status": "max_iterations_reached"}

        return final_state
