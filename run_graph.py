from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import time
import operator
from rich.console import Console
from rich.table import Table

# ============= STATE DEFINITION =============

class InputText(TypedDict):
    language: str
    text: str

class RawArticle(TypedDict):
    article_id: int
    language: str
    text: str

class TranslatedArticles(TypedDict):
    article_id: int
    source_language: str
    text_en: str

class ModelResult(TypedDict):
    article_id: int
    source_language: str
    model: str
    score: float

class GraphState(TypedDict):
    selected_languages: list[str]
    num_articles: int
    input_text: list[InputText]
    raw_articles: list[RawArticle]
    translated_articles: list[TranslatedArticles]
    results: Annotated[list[ModelResult], operator.add]
    summary: str

# ============= NODES =============

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

# ============= GRAPH CONSTRUCTION =============

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("translate", translate_node)
    workflow.set_entry_point("translate")
    return workflow.compile()

# ============= EXECUTION =============

if __name__ == "__main__":
    import argparse
    import sys

    try:
        import rich
    except ImportError:
        print("❗ The 'rich' library is not installed. Please run: pip install rich")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="🌍 Cross-Lingual Sentiment Analyzer (LangGraph Demo)"
    )
    parser.add_argument(
        "--text", "-t",
        required=True,
        help="Input text for translation and sentiment analysis."
    )
    parser.add_argument(
        "--langs", "-l",
        required=True,
        help="Comma-separated list of target languages, e.g., en,pl,de,fr"
    )
    parser.add_argument(
        "--articles", "-a",
        type=int,
        default=3,
        help="Number of articles to scrape per language (default: 3)."
    )
    args = parser.parse_args()

    selected_languages = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    if not selected_languages:
        print("❗ No languages provided. Example usage: --langs en,pl,de")
        sys.exit(1)

    graph = build_graph()

    initial_state = {
        "input_text": args.text,
        "selected_languages": selected_languages,
        "translated_texts": {},
        "scraped_articles": {},
        "translated_to_english": {},
        "results": [],
        "summary": "",
        "num_articles": args.articles
    }

    print("=" * 70)
    print("🚀 Starting Cross-Lingual Sentiment Analyzer (LangGraph + rich)")
    print(f"📝 Input text: {args.text}")
    print(f"🌎 Languages: {', '.join(selected_languages)}")
    print(f"📰 Articles per language: {args.articles}")
    print("=" * 70)

    start_time = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"✅ Done in {elapsed:.2f}s")
    print(f"📊 Total results: {len(final_state.get('results', []))}")
    print(f"🧾 Summary: {final_state.get('summary', 'N/A')}")
    print("=" * 70)
