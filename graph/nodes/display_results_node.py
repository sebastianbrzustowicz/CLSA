from graph.state_definitions import GraphState, ModelResult
from typing import Dict, Tuple
from collections import defaultdict
from tabulate import tabulate
import os
import html
import re
from datetime import datetime

MODEL_SHORT_NAMES = {
    "models/encoders/twitter-roberta-base-sentiment-latest": "sentiment",
    "unitary/toxic-bert": "toxicity",
    "j-hartmann/emotion-english-distilroberta-base": "emotion",
    "cardiffnlp/twitter-roberta-base-irony": "irony",
    "cointegrated/roberta-base-formality": "formality",
    "GroNLP/mdebertav3-subjectivity-english": "objectivity",
    "IDA-SERICS/PropagandaDetection": "propaganda"
}

FIXED_LABELS = {
    "formality": "formal",
    "irony": "irony",
    "propaganda": "propaganda",
    "objectivity": "objective"
}

HTML_COLORS = {
    "positive": "green",
    "neutral": "black",
    "negative": "red",
    "anger": "red",
    "fear": "darkred",
    "disgust": "darkred",
    "sadness": "red",
    "joy": "green",
    "surprise": "green"
}

def colorize_cell_cli(model: str, labels: list[Tuple[str, float]]) -> str:
    reset = "\033[0m"
    if model in ["sentiment", "emotion"]:
        colored_labels = []
        for i, (lbl, val) in enumerate(labels):
            if model == "sentiment":
                colors = {"positive": "\033[92m", "neutral": "\033[97m", "negative": "\033[91m"}
            else:
                colors = {"anger":"\033[91m", "fear":"\033[91m", "disgust":"\033[91m",
                          "sadness":"\033[91m", "neutral":"\033[97m", "joy":"\033[92m", "surprise":"\033[92m"}
            col = colors.get(lbl, "\033[97m") if i == 0 else "\033[90m"
            colored_labels.append(f"{col}{lbl} ({val:.3f}){reset}")
        return ", ".join(colored_labels)
    else:
        label, value = labels[0] if labels else ("", 0.0)
        if model == "toxicity":
            color = "\033[92m" if value < 0.33 else ("\033[97m" if value < 0.66 else "\033[91m")
        elif model in ["objectivity", "formality"]:
            color = "\033[91m" if value < 0.33 else ("\033[97m" if value < 0.66 else "\033[92m")
        elif model in ["propaganda", "irony"]:
            color = "\033[92m" if value < 0.33 else ("\033[97m" if value < 0.66 else "\033[91m")
        else:
            color = "\033[97m"
        return f"{color}{value:.3f}{reset}"

def colorize_cell_html(model: str, labels: list[Tuple[str, float]]) -> str:
    if model in ["sentiment", "emotion"]:
        colored_labels = []
        for i, (lbl, val) in enumerate(labels):
            color = HTML_COLORS.get(lbl, "black") if i == 0 else "gray"
            colored_labels.append(f'<span style="color:{color}">{html.escape(lbl)} ({val:.3f})</span>')
        return ", ".join(colored_labels)
    else:
        value = labels[0][1] if labels else 0.0
        if model == "toxicity":
            color = "green" if value < 0.33 else ("black" if value < 0.66 else "red")
        elif model in ["objectivity", "formality"]:
            color = "red" if value < 0.33 else ("black" if value < 0.66 else "green")
        elif model in ["propaganda", "irony"]:
            color = "green" if value < 0.33 else ("black" if value < 0.66 else "red")
        else:
            color = "black"
        return f'<span style="color:{color}">{value:.3f}</span>'

def display_results_node(state: GraphState) -> GraphState:
    results: list[ModelResult] = state.get("results", [])
    if not results:
        print("❗ No results to display.")
        return {}

    # --- Results aggregation ---
    aggregated: Dict[str, Dict[str, list[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        lang = r["source_language"]
        model_short = MODEL_SHORT_NAMES.get(r["model"], r["model"])
        score = r["score"]

        if isinstance(score, dict):
            if model_short in FIXED_LABELS:
                val = float(score.get(FIXED_LABELS[model_short], 0.0))
                aggregated[lang][model_short].append((FIXED_LABELS[model_short], val))
            else:
                for lbl, val in score.items():
                    aggregated[lang][model_short].append((lbl, float(val)))
        else:
            aggregated[lang][model_short].append(("", float(score)))

    # --- Calculate mean and sort ---
    final_data = {}
    for lang, model_data in aggregated.items():
        final_data[lang] = {}
        for model, vals in model_data.items():
            if model in FIXED_LABELS:
                avg_value = sum(v for _, v in vals) / len(vals)
                final_data[lang][model] = [(FIXED_LABELS[model], avg_value)]
            else:
                label_avg = {lbl: sum(v for l,v in vals if l==lbl)/len([v for l,v in vals if l==lbl]) for lbl,_ in vals}
                if model == "sentiment":
                    sorted_labels = sorted(label_avg.items(), key=lambda x: x[1], reverse=True)
                else:
                    sorted_labels = sorted(label_avg.items(), key=lambda x: x[1], reverse=True)[:2]
                final_data[lang][model] = sorted_labels

    all_models = ["emotion", "formality", "irony", "propaganda", "sentiment", "objectivity", "toxicity"]

    # --- Display in CLI ---
    table_rows_cli = []
    for lang, model_scores in final_data.items():
        row = [lang]
        for model in all_models:
            if model in model_scores:
                row.append(colorize_cell_cli(model, model_scores[model]))
            else:
                row.append("-")
        table_rows_cli.append(row)

    headers = ["Language", "emotion", "formality", "irony", "propaganda", "sentiment", "objectivity", "toxicity"]
    print("\n📊 ADVANCED COLOR-CODED RESULTS TABLE (CLI view):\n")
    print(tabulate(table_rows_cli, headers=headers, tablefmt="fancy_grid"))

    # --- Create HTML ---
    html_rows = []
    for lang, model_scores in final_data.items():
        row_html = f"<tr><td>{html.escape(lang)}</td>"
        for model in all_models:
            if model in model_scores:
                row_html += f"<td>{colorize_cell_html(model, model_scores[model])}</td>"
            else:
                row_html += "<td>-</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    prompt = next((t["text"] for t in state.get("input_text", []) if t["language"] == "en"), "CLS Analyzer Results")

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    html_table = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>CLSA</title>
    <style>
    table {{border-collapse: collapse; width: 100%;}}
    td, th {{border: 1px solid black; padding: 5px; text-align: center;}}
    </style>
    </head>
    <body>
    <h2>Cross-lingual Sentiment Analyzer</h2>
    <p style="font-size:small;"><strong>Prompt:</strong> {html.escape(prompt)}</p>
    <p style="font-size:small;"><strong>Date:</strong> {current_time}</p>
    <table>
        <tr>
            <th>Language</th><th>emotion</th><th>formality</th><th>irony</th>
            <th>propaganda</th><th>sentiment</th><th>objectivity</th><th>toxicity</th>
        </tr>
        {''.join(html_rows)}
    </table>
    </body>
    </html>
    """

    os.makedirs("output", exist_ok=True)
    output_file = "output/results.html"
    safe_prompt = re.sub(r'[\\/:"*?<>|]+', "_", prompt)
    safe_prompt = safe_prompt.replace(" ", "-")
    output_file = f"output/clsa_results_{safe_prompt}.html"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_table)

    print(f"\n✅ Table saved to {output_file} with HTML view (colored).")