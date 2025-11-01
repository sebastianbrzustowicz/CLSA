import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from graph.state_definitions import GraphState, RawArticle, TranslatedArticles
from typing import List
import re

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sentences if s]

def translate_to_en_node(state: GraphState) -> GraphState:
    print("\n🌍 NODE: translate_articles_node (IMPROVED, NO REPETITION)")

    raw_articles: list[RawArticle] = state.get("raw_articles", [])
    if not raw_articles:
        print("❗ No raw articles to translate. Skipping.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "models/translation/m2m100_418M"

    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    translated_entries: list[TranslatedArticles] = []
    existing_ids = {a["article_id"] for a in state.get("translated_articles", [])}

    for idx, article in enumerate(raw_articles):
        article_id = article["article_id"]
        source_lang = article["language"]
        text = article["text"]

        if article_id in existing_ids:
            continue
        if source_lang == "en":
            translated_entries.append({
                "article_id": article_id,
                "source_language": "en",
                "text_en": text
            })
            continue

        tokenizer.src_lang = source_lang
        sentences = split_into_sentences(text)

        translated_parts = []
        for sentence in sentences:
            encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id("en"),
                    max_new_tokens=min(128, encoded.input_ids.shape[1]*2),
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                    early_stopping=True
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated_parts.append(decoded[0])

        translated_text = " ".join(translated_parts)
        translated_entries.append({
            "article_id": article_id,
            "source_language": source_lang,
            "text_en": translated_text
        })

        print(f"✅ [{source_lang.upper()}] Article {article_id} translated ({idx+1}/{len(raw_articles)})")

    return {"translated_articles": translated_entries}
