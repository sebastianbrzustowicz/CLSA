import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from graph.state_definitions import GraphState, TranslatedArticles, ModelResult
from typing import List, Dict
import torch.nn.functional as F

def emotion_node(state: GraphState) -> GraphState:
    """
    Emotion analysis node using 'j-hartmann/emotion-english-distilroberta-base'.
    Handles long texts by splitting them into chunks.
    """
    print("\n📝 NODE: emotion_node")

    translated_articles: List[TranslatedArticles] = state.get("translated_articles", [])
    if not translated_articles:
        print("❗ No translated articles found. Skipping emotion analysis.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🖥 Using device: {device}")

    model_path = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device, non_blocking=True)

    max_tokens = 128  # chunk size to stay within 512-token limit
    class_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    # IDs of articles already processed by this model
    existing_results = state.get("results", [])
    existing_ids_for_model = {
        r["article_id"] for r in existing_results if r.get("model") == model_path
    }

    new_results: List[ModelResult] = []

    for article in translated_articles:
        if article["article_id"] in existing_ids_for_model:
            continue  # skip already processed article

        text = article["text_en"]

        # Tokenize full text and split into smaller chunks
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
                probs = F.softmax(logits, dim=-1)
                all_scores.append(probs.cpu())

        # Average probabilities across all chunks
        avg_scores = torch.mean(torch.stack(all_scores), dim=0).squeeze(0)
        score_dict: Dict[str, float] = {cls: float(avg_scores[i]) for i, cls in enumerate(class_labels)}

        result: ModelResult = {
            "article_id": article["article_id"],
            "source_language": article["source_language"],
            "model": model_path,
            "score": score_dict
        }
        new_results.append(result)
        print(f"[{article['article_id']}] Emotion analyzed: {score_dict}")

    return {"results": new_results}
