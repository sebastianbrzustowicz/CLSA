from rich import print_json
from graph.state_definitions import GraphState

def print_final_state_node(state: GraphState) -> GraphState:
    """
    Prints the final workflow state in a structured JSON format.
    """
    print("\n🧩 NODE: print_final_state_node — displaying final GraphState\n")
    try:
        print_json(data=state)
    except Exception as e:
        print(f"⚠️ Could not pretty print state: {e}")
        print(state)
    return state
