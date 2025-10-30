from langgraph.graph import StateGraph, END
from graph.state_definitions import GraphState
from graph.nodes.translate_to_multiple_node import translate_to_multiple_node
from graph.nodes.print_final_state_node import print_final_state_node
from graph.nodes.scrape_node import scrape_node_factory

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

    # --- Final print node ---
    workflow.add_node("print_final_state", print_final_state_node)
    for node_name in scrape_nodes:
        workflow.add_edge(node_name, "print_final_state")

    # --- End of workflow ---
    workflow.add_edge("print_final_state", END)

    # Entry point
    workflow.set_entry_point("translate_to_many")

    return workflow.compile()
