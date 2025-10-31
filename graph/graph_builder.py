from graph.nodes.scrape_node import scrape_node_factory
from graph.nodes.save_final_state_node import save_final_state_node
from graph.nodes.translate_to_multiple_node import translate_to_multiple_node
from graph.nodes.translate_to_en_node import translate_to_en_node
from graph.state_definitions import GraphState
from langgraph.graph import StateGraph, END

# --- Encoder models ---
from graph.nodes.sentiment.sentiment_cardiff_node import sentiment_cardiff_node
from graph.nodes.sentiment.toxic_bert_node import toxic_bert_node
from graph.nodes.sentiment.emotion_node import emotion_node
from graph.nodes.sentiment.irony_node import irony_node
from graph.nodes.sentiment.formality_node import formality_node
from graph.nodes.sentiment.subjectivity_node import subjectivity_node
from graph.nodes.sentiment.propaganda_detection_node import propaganda_detection_node

from graph.nodes.display_results_node import display_results_node


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

    # --- Translate articles to english ---
    workflow.add_node("translate_articles", translate_to_en_node)
    for node_name in scrape_nodes:
        workflow.add_edge(node_name, "translate_articles")

    # --- Analyze sentiment ---
    previous_node = "translate_articles"

    workflow.add_node("sentiment_cardiff", sentiment_cardiff_node)
    workflow.add_edge(previous_node, "sentiment_cardiff")
    previous_node = "sentiment_cardiff"

    workflow.add_node("toxic_bert", toxic_bert_node)
    workflow.add_edge(previous_node, "toxic_bert")
    previous_node = "toxic_bert"

    workflow.add_node("emotion_analysis", emotion_node)
    workflow.add_edge(previous_node, "emotion_analysis")
    previous_node = "emotion_analysis"

    workflow.add_node("irony_analysis", irony_node)
    workflow.add_edge(previous_node, "irony_analysis")
    previous_node = "irony_analysis"

    workflow.add_node("formality_analysis", formality_node)
    workflow.add_edge(previous_node, "formality_analysis")
    previous_node = "formality_analysis"

    workflow.add_node("subjectivity_mdeberta", subjectivity_node)
    workflow.add_edge(previous_node, "subjectivity_mdeberta")
    previous_node = "subjectivity_mdeberta"

    workflow.add_node("propaganda_detection", propaganda_detection_node)
    workflow.add_edge(previous_node, "propaganda_detection")
    previous_node = "propaganda_detection"

    # --- Save state ---
    workflow.add_node("save_final_state_node", save_final_state_node)
    workflow.add_edge(previous_node, "save_final_state_node")

    # --- Display results in the table ---
    workflow.add_node("display_results", display_results_node)
    workflow.add_edge("save_final_state_node", "display_results")

    # --- End ---
    workflow.add_edge("display_results", END)
    workflow.set_entry_point("translate_to_many")

    return workflow.compile()
