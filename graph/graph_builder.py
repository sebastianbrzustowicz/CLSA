from graph.nodes.scrape_node import scrape_node_factory
from graph.nodes.save_final_state_node import save_final_state_node
from graph.nodes.translate_to_multiple_node import translate_to_multiple_node
from graph.state_definitions import GraphState
from langgraph.graph import StateGraph, END
from graph.nodes.translate_to_en_node import translate_to_en_node

def build_graph(initial_state: GraphState):
    workflow = StateGraph(GraphState)

    # --- Entry node ---
    workflow.add_node("translate_to_many", translate_to_multiple_node)

    # --- Scraping nodes ---
    scrape_nodes = []
    for lang in initial_state["selected_languages"]:
        node_name = f"scrape_{lang}"
        workflow.add_node(node_name, scrape_node_factory(lang))
        workflow.add_edge("translate_to_many", node_name)
        scrape_nodes.append(node_name)

    # --- Translate articles node ---
    workflow.add_node("translate_articles", translate_to_en_node)
    for node_name in scrape_nodes:
        workflow.add_edge(node_name, "translate_articles")

    # --- Final print node ---
    workflow.add_node("save_final_state_node", save_final_state_node)
    workflow.add_edge("translate_articles", "save_final_state_node")

    # --- End of workflow ---
    workflow.add_edge("save_final_state_node", END)

    # Entry point
    workflow.set_entry_point("translate_to_many")

    return workflow.compile()
