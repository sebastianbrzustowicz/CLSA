from langgraph.graph import StateGraph, END
from graph.state_definitions import GraphState
from graph.nodes.nodes import translate_node

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("translate", translate_node)
    workflow.set_entry_point("translate")
    return workflow.compile()
