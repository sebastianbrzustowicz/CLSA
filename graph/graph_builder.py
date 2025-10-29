from langgraph.graph import StateGraph, END
from graph.state_definitions import GraphState
from graph.nodes.translate_to_multiple_node import translate_to_multiple_node

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("translate", translate_to_multiple_node)
    """
    languages = ["pl", "de", "fr"]
    for lang in languages:
        workflow.add_node(f"scrape_{lang}", scrape_node)
        workflow.add_edge("translate", f"scrape_{lang}")
      
    workflow.add_node("translate_to_en", translate_to_english_node)
    for lang in languages:
        workflow.add_edge(f"scrape_{lang}", "translate_to_en")
    
    models = ["model1", "model2", "model3", "model4", "model5", "model5"]
    for model_node in models:
        workflow.add_node(model_node, globals()[model_node + "_node"])
        workflow.add_edge("translate_to_en", model_node)
    
    workflow.add_node("summarize", summarize_node)
    for model_node in models:
        workflow.add_edge(model_node, "summarize")
    
    workflow.add_node("display_table", display_table_node)
    workflow.add_edge("summarize", "display_table")
    workflow.add_edge("display_table", END)
    """
    workflow.set_entry_point("translate")
    return workflow.compile()
