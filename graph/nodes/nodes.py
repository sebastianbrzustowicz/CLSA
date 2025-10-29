import time
from graph.state_definitions import GraphState

def translate_node(state: GraphState) -> GraphState:
    """
    Example translation node (dummy implementation)
    """
    print(f"\n🌍 NODE: translate_node")
    print(f"   Input: '{state['input_text']}'")
    print(f"   Languages: {state['selected_languages']}")

    translated = {}

    for language in state["selected_languages"]:
        time.sleep(0.05)
        translated[language] = f"{state['input_text']} [translated to {language}]"
        print(f"   ✓ Translated to {language}")

    return {**state, "translated_texts": translated}
