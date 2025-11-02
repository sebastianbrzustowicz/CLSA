import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from graph.state_definitions import GraphState, TranslatedArticles, ModelResult
from typing import List, Dict
import torch.nn.functional as F

def irony_node(state: GraphState, debug: bool = False) -> GraphState:
    """
    Node that analyzes irony using 'cardiffnlp/twitter-roberta-base-irony'.
    Handles long texts automatically with tokenizer overflow chunks.
    The score field contains the full probability distribution for the two classes: irony, non_irony.
    """
    print("\n📝 NODE: irony_node")

    translated_articles: List[TranslatedArticles] = state.get("translated_articles", [])
    if not translated_articles:
        if debug:
            print("❗ No translated articles found. Skipping irony analysis.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        print(f"   🖥 Using device: {device}")

    model_path = "cardiffnlp/twitter-roberta-base-irony"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(device)

    max_length = 512
    class_labels = ["irony", "non_irony"]

    existing_results = state.get("results", [])
    existing_ids_for_model = {r["article_id"] for r in existing_results if r.get("model") == model_path}

    new_results: List[ModelResult] = []

    total = len(translated_articles)
    processed_count = 0

    for article in translated_articles:
        if article["article_id"] in existing_ids_for_model:
            processed_count += 1
            if debug:
                percent = (processed_count / total) * 100
                print(f"\rProgress: {percent:.1f}% ({processed_count}/{total})", end="", flush=True)
            continue

        text = article.get("text_en", "")
        if not text.strip():
            processed_count += 1
            if debug:
                percent = (processed_count / total) * 100
                print(f"\rProgress: {percent:.1f}% ({processed_count}/{total})", end="", flush=True)
            continue

        encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=50,
            return_overflowing_tokens=True,
            padding=False
        )

        all_scores = []
        for input_ids, attention_mask in zip(encodings["input_ids"], encodings["attention_mask"]):
            inputs = tokenizer.pad(
                {"input_ids": [input_ids], "attention_mask": [attention_mask]},
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                all_scores.append(probs.cpu())

        avg_scores = torch.mean(torch.stack(all_scores), dim=0).squeeze(0)
        score_dict: Dict[str, float] = {cls: float(avg_scores[i]) for i, cls in enumerate(class_labels)}

        new_results.append({
            "article_id": article["article_id"],
            "source_language": article.get("source_language", "unknown"),
            "model": model_path,
            "score": score_dict
        })

        processed_count += 1
        percent = (processed_count / total) * 100
        print(f"\rProgress: {percent:.1f}% ({processed_count}/{total})", end="", flush=True)

    if debug:
        print()

    return {"results": new_results}
