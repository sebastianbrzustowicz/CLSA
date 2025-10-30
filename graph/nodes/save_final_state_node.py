import json
import os
from graph.state_definitions import GraphState

def save_final_state_node(state: GraphState) -> GraphState:
    """
    Saves the final workflow state as a JSON file in the project directory.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_state.json")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"\n✅ NODE: save_final_state_node — state saved to '{output_file}'\n")
    except Exception as e:
        print(f"⚠️ Could not save state to JSON: {e}")

    return state
