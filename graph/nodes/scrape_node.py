import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse, parse_qs, unquote
from graph.state_definitions import GraphState, RawArticle

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

def scrape_node_factory(language: str, min_length: int = 150):
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
        params = {"q": f"{query_text} news", "kl": "wt-wt", "s": "0"}

        print(f"[{language.upper()}] 🔍 Searching DuckDuckGo news for: '{query_text[:50]}...'")

        session = requests.Session()

        def safe_get(url, params=None, retry_delay=5):
            headers = {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://duckduckgo.com/"}
            try:
                resp = session.get(url, params=params, headers=headers, timeout=10)
                if resp.status_code == 403:
                    print(f"[{language.upper()}] 🚫 DDG 403 Forbidden — retrying after {retry_delay}s...")
                    time.sleep(retry_delay)
                    return None
                resp.raise_for_status()
                return resp
            except Exception as e:
                print(f"[{language.upper()}] ❗ Request failed: {e}")
                return None

        page = 0
        max_pages = 5

        # DuckDuckGo phase
        while len(collected) < num_articles and page < max_pages:
            params["s"] = str(page * 50)
            resp = safe_get(base_url, params)
            if not resp:
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
                    headers = {"User-Agent": random.choice(USER_AGENTS)}
                    r = requests.get(actual_url, headers=headers, timeout=10)
                    r.encoding = r.apparent_encoding
                    r.raise_for_status()
                    page_soup = BeautifulSoup(r.text, "html.parser")

                    for tag in page_soup(["script", "style", "noscript", "header", "footer", "aside", "form", "nav"]):
                        tag.decompose()

                    paragraphs = [p.get_text(" ", strip=True) for p in page_soup.find_all("p")]
                    text_body = "\n".join(p for p in paragraphs if len(p) > 50)

                    if not text_body or len(text_body) < min_length:
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

                time.sleep(random.uniform(1.0, 2.5))

            page += 1

        # Fallback if too few collected — Bing News RSS
        if len(collected) < num_articles:
            missing = num_articles - len(collected)
            print(f"\n[{language.upper()}] ⚠️ Only {len(collected)} collected — using Bing News RSS to fetch {missing} more...")
            bing_rss = f"https://www.bing.com/news/search?q={query_text}&format=rss"

            try:
                resp = requests.get(bing_rss, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "xml")
                items = soup.find_all("item")

                for item in items:
                    if len(collected) >= num_articles:
                        break
                    link = item.link.text.strip()
                    if link in visited_urls:
                        continue
                    visited_urls.add(link)

                    try:
                        r = requests.get(link, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=10)
                        r.encoding = r.apparent_encoding
                        r.raise_for_status()
                        ps = BeautifulSoup(r.text, "html.parser")
                        for tag in ps(["script", "style", "noscript", "header", "footer", "aside", "form", "nav"]):
                            tag.decompose()
                        paragraphs = [p.get_text(" ", strip=True) for p in ps.find_all("p")]
                        text_body = "\n".join(p for p in paragraphs if len(p) > 50)
                        if not text_body or len(text_body) < min_length:
                            continue
                        collected.append({
                            "article_id": start_article_id + len(collected),
                            "language": language,
                            "text": text_body
                        })
                    except Exception:
                        continue

            except Exception as e:
                print(f"[{language.upper()}] ❗ Bing News RSS failed: {e}")

        # Final report
        if len(collected) >= num_articles:
            print(f"\n[{language.upper()}] 🎉 Finished! Collected {len(collected)} articles ✅")
        else:
            print(f"\n[{language.upper()}] ⚠️ Finished but only {len(collected)}/{num_articles} collected ❌")

        new_raw_articles = state.get("raw_articles", []) + collected
        return {"raw_articles": new_raw_articles}

    return scrape_node
