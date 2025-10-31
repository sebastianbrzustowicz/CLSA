import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from graph.state_definitions import GraphState, TranslatedArticles, ModelResult
from typing import List, Dict
import torch.nn.functional as F

def formality_node(state: GraphState) -> GraphState:
    """
    Formality analysis node using 'cointegrated/roberta-base-formality'.
    """
    print("\n📝 NODE: formality_node")

    translated_articles: List[TranslatedArticles] = state.get("translated_articles", [])
    if not translated_articles:
        print("❗ No translated articles found. Skipping formality analysis.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🖥 Using device: {device}")

    model_path = "cointegrated/roberta-base-formality"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device, non_blocking=True)

    max_tokens = 128
    class_labels = ["formal", "informal"]

    existing_results = state.get("results", [])
    existing_ids_for_model = {r["article_id"] for r in existing_results if r.get("model") == model_path}

    new_results: List[ModelResult] = []

    for article in translated_articles:
        if article["article_id"] in existing_ids_for_model:
            continue

        text = article["text_en"]
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

        all_scores = []
        for chunk_tokens in chunks:
            chunk_tensor = torch.tensor([chunk_tokens]).to(device)
            with torch.no_grad():
                logits = model(chunk_tensor).logits
                probs = F.softmax(logits, dim=-1)
                all_scores.append(probs.cpu())

        avg_scores = torch.mean(torch.stack(all_scores), dim=0).squeeze(0)
        score_dict: Dict[str, float] = {cls: float(avg_scores[i]) for i, cls in enumerate(class_labels)}

        new_results.append({
            "article_id": article["article_id"],
            "source_language": article["source_language"],
            "model": model_path,
            "score": score_dict
        })

        print(f"[{article['article_id']}] Formality analyzed: {score_dict}")

    return {"results": new_results}
