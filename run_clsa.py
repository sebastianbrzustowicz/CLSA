import argparse
import sys
import time
from graph.graph_builder import build_graph

def main():
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

if __name__ == "__main__":
    main()
