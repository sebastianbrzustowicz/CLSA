import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from graph.state_definitions import GraphState, TranslatedArticles, ModelResult
from typing import List, Dict
import torch.nn.functional as F

def toxic_bert_node(state: GraphState) -> GraphState:
    """
    Node that analyzes toxicity using 'unitary/toxic-bert'.
    """
    print("\n📝 NODE: toxic_bert_node")

    translated_articles: List[TranslatedArticles] = state.get("translated_articles", [])
    if not translated_articles:
        print("❗ No translated articles found. Skipping toxicity analysis.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🖥 Using device: {device}")

    model_path = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device, non_blocking=True)

    max_tokens = 128
    class_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    existing_ids_for_model = {
        r["article_id"] for r in state.get("results", []) if r.get("model") == model_path
    }

    new_results: List[ModelResult] = []

    for article in translated_articles:
        if article["article_id"] in existing_ids_for_model:
            continue

        text = article["text_en"]

        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

        all_scores = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            encoded = tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=model.config.max_position_embeddings
            ).to(device)

            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.sigmoid(logits)
                all_scores.append(probs.cpu())

        avg_scores = torch.mean(torch.stack(all_scores), dim=0).squeeze(0)
        score_dict: Dict[str, float] = {cls: float(avg_scores[i]) for i, cls in enumerate(class_labels)}

        result: ModelResult = {
            "article_id": article["article_id"],
            "source_language": article["source_language"],
            "model": model_path,
            "score": score_dict
        }
        new_results.append(result)
        print(f"[{article['article_id']}] Toxicity analyzed: {score_dict}")

    return {"results": new_results}
