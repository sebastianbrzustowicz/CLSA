from graph.state_definitions import GraphState, ModelResult
from typing import Dict, Tuple
from collections import defaultdict
from tabulate import tabulate

MODEL_SHORT_NAMES = {
    "models/encoders/twitter-roberta-base-sentiment-latest": "sentiment",
    "unitary/toxic-bert": "toxicity",
    "j-hartmann/emotion-english-distilroberta-base": "emotion",
    "cardiffnlp/twitter-roberta-base-irony": "irony",
    "cointegrated/roberta-base-formality": "formality",
    "GroNLP/mdebertav3-subjectivity-english": "objectivity",
    "IDA-SERICS/PropagandaDetection": "propaganda"
}

# Fixed labels for single-valued models
FIXED_LABELS = {
    "formality": "formal",
    "irony": "irony",
    "propaganda": "propaganda",
    "objectivity": "objective"
}

def colorize_cell(model: str, label: str, value: float) -> str:
    reset = "\033[0m"
    if model == "sentiment":
        color = "\033[92m" if label == "positive" else ("\033[91m" if label == "negative" else "\033[97m")
        return f"{color}{label} ({value:.3f}){reset}"
    elif model == "toxicity":
        color = "\033[92m" if value < 0.33 else ("\033[97m" if value < 0.66 else "\033[91m")
        return f"{color}{label} ({value:.3f}){reset}"
    elif model in ["objectivity", "formality"]:
        color = "\033[91m" if value < 0.33 else ("\033[97m" if value < 0.66 else "\033[92m")
        return f"{color}{value:.3f}{reset}"
    elif model in ["propaganda", "irony"]:
        color = "\033[92m" if value < 0.33 else ("\033[97m" if value < 0.66 else "\033[91m")
        return f"{color}{value:.3f}{reset}"
    elif model == "emotion":
        if label in ["anger", "fear", "disgust", "sadness"]:
            color = "\033[91m"
        elif label == "neutral":
            color = "\033[97m"
        else:
            color = "\033[92m"
        return f"{color}{label} ({value:.3f}){reset}"
    else:
        return f"{value:.3f}"

def display_results_node(state: GraphState) -> GraphState:
    results: list[ModelResult] = state.get("results", [])
    if not results:
        print("❗ No results to display.")
        return {}

    # Aggregate results by language and model
    aggregated: Dict[str, Dict[str, list[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        lang = r["source_language"]
        model_short = MODEL_SHORT_NAMES.get(r["model"], r["model"])
        score = r["score"]

        if isinstance(score, dict):
            if model_short in FIXED_LABELS:
                fixed_label = FIXED_LABELS[model_short]
                val = float(score.get(fixed_label, 0.0))
                aggregated[lang][model_short].append(("", val))
            else:  # sentiment or emotion
                for lbl, val in score.items():
                    aggregated[lang][model_short].append((lbl, float(val)))
        else:
            aggregated[lang][model_short].append(("", float(score)))

    # Compute averages and select the dominant label
    final_data = {}
    for lang, model_data in aggregated.items():
        final_data[lang] = {}
        for model, vals in model_data.items():
            if model in FIXED_LABELS:
                values = [v for _, v in vals]
                avg_value = sum(values) / len(values)
                final_data[lang][model] = ("", avg_value)
            else:
                label_sums: Dict[str, list[float]] = defaultdict(list)
                for lbl, val in vals:
                    label_sums[lbl].append(val)
                label_avg = {lbl: sum(lst) / len(lst) for lbl, lst in label_sums.items()}
                max_label = max(label_avg, key=label_avg.get)
                final_data[lang][model] = (max_label, label_avg[max_label])

    all_models = ["emotion", "formality", "irony", "propaganda", "sentiment", "objectivity", "toxicity"]

    table_rows = []
    for lang, model_scores in final_data.items():
        row = [lang]
        for model in all_models:
            if model in model_scores:
                label, val = model_scores[model]
                row.append(colorize_cell(model, label, val))
            else:
                row.append("-")
        table_rows.append(row)

    headers = ["Language", "emotion", "formality", "irony", "propaganda", "sentiment", "objectivity", "toxicity"]

    print("\n📊 ADVANCED COLOR-CODED RESULTS TABLE (interpreted labels + color):\n")
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

    return {}
