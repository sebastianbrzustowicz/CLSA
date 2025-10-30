import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, parse_qs, unquote
from graph.state_definitions import GraphState, RawArticle

def scrape_node_factory(language: str):
    def scrape_node(state: GraphState) -> GraphState:
        candidates = [it for it in state["input_text"] if it["language"] == language]
        if not candidates:
            print(f"[{language.upper()}] ❗ No input_text entry found — skipping.")
            return {}

        query_text = candidates[0]["text"]
        num_articles = state.get("num_articles", 3)
        collected = []
        start_article_id = len(state.get("raw_articles", []))
        visited_urls = set()

        base_url = "https://duckduckgo.com/html/"
        params = {"q": f"{query_text} news", "kl": f"{language}-en", "s": "0"}

        print(f"[{language.upper()}] 🔍 Searching DuckDuckGo news for: '{query_text[:50]}...'")

        page = 0
        max_pages = 5
        while len(collected) < num_articles and page < max_pages:
            params["s"] = str(page * 50)
            try:
                resp = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                resp.raise_for_status()
            except Exception as e:
                print(f"[{language.upper()}] ❗ DDG request failed: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            links = [a.get("href") for a in soup.select("a.result__a, a.result__url") if a.get("href")]
            if not links:
                break

            print(f"[{language.upper()}] 🌐 Found {len(links)} potential results on page {page + 1}")

            for link in links:
                if len(collected) >= num_articles:
                    break

                parsed = urlparse(link)
                qs = parse_qs(parsed.query)
                actual_url = unquote(qs["uddg"][0]) if "uddg" in qs else link

                if actual_url in visited_urls:
                    continue
                visited_urls.add(actual_url)

                try:
                    r = requests.get(actual_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    r.encoding = r.apparent_encoding
                    r.raise_for_status()
                    page_soup = BeautifulSoup(r.text, "html.parser")

                    for tag in page_soup(["script", "style", "noscript", "header", "footer", "aside", "form", "nav"]):
                        tag.decompose()

                    paragraphs = [p.get_text(" ", strip=True) for p in page_soup.find_all("p")]
                    text_body = "\n".join(p for p in paragraphs if len(p) > 50)

                    if not text_body:
                        continue
                    if language != "en" and any(x in text_body.lower() for x in ["cookies", "privacy", "accept", "terms", "javascript"]):
                        continue

                    article_entry: RawArticle = {
                        "article_id": start_article_id + len(collected),
                        "language": language,
                        "text": text_body
                    }
                    collected.append(article_entry)

                    print(f"[{language.upper()}] ✅ Articles collected: {len(collected)}/{num_articles}", end="\r")

                except Exception:
                    continue

                time.sleep(0.5)

            page += 1

        if collected:
            print(f"\n[{language.upper()}] 🎉 Finished! Collected {len(collected)} articles.")
        else:
            print(f"\n[{language.upper()}] ⚠️ No valid articles collected.")

        new_raw_articles = state.get("raw_articles", []) + collected
        return {"raw_articles": new_raw_articles}

    return scrape_node
