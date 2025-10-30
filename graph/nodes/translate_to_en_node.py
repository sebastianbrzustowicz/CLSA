import torch
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from graph.state_definitions import GraphState, RawArticle, TranslatedArticles
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter based on punctuation."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sentences if s]

def translate_to_en_node(state: GraphState) -> GraphState:
    """
    Node that translates all RawArticles into English using local M2M100 model.
    Splits text into small safe chunks (~150 tokens) to preserve translation quality.
    Each article is translated independently to avoid context mixing.
    """
    print("\n🌍 NODE: translate_articles_node")

    raw_articles: List[RawArticle] = state.get("raw_articles", [])
    if not raw_articles:
        print("❗ No raw articles found to translate. Skipping.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🖥 Using device: {device}")

    model_path = "models/translation/m2m100_418M"
    translated_entries: List[TranslatedArticles] = []
    existing_ids = {a["article_id"] for a in state.get("translated_articles", [])}

    for idx, article in enumerate(raw_articles):
        article_id = article["article_id"]
        source_lang = article["language"]
        source_text = article["text"]

        if article_id in existing_ids:
            print(f"[{source_lang.upper()}] Article {article_id} already translated. Skipping.")
            continue

        if source_lang == "en":
            translated_entries.append({
                "article_id": article_id,
                "source_language": "en",
                "text_en": source_text
            })
            continue

        # Independent model + tokenizer per article to avoid state carryover
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer.src_lang = source_lang

        # Split text into sentences
        sentences = split_into_sentences(source_text)
        translated_parts = []

        for sent_idx, sentence in enumerate(sentences):
            tokens = tokenizer.tokenize(sentence)
            max_tokens = 150
            chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]

            chunk_translations = []
            for chunk_idx, chunk_tokens in enumerate(chunks):
                chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                encoded = tokenizer(chunk_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        forced_bos_token_id=tokenizer.get_lang_id("en"),
                        max_new_tokens=256
                    )

                part_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                chunk_translations.append(part_text)
                print(f"[{source_lang.upper()}] Article {article_id}, sent {sent_idx+1}/{len(sentences)}, chunk {chunk_idx+1}/{len(chunks)} translated.")

            translated_sentence = " ".join(chunk_translations)
            translated_parts.append(translated_sentence)

        translated_text = " ".join(translated_parts)

        translated_entries.append({
            "article_id": article_id,
            "source_language": source_lang,
            "text_en": translated_text
        })

        print(f"✅ [{source_lang.upper()}] Article {article_id} fully translated ({idx+1}/{len(raw_articles)})")

        # Free GPU memory between articles
        del tokenizer, model
        torch.cuda.empty_cache()

    return {"translated_articles": translated_entries}
