from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from graph.state_definitions import GraphState, InputText

def translate_to_multiple_node(state: GraphState) -> GraphState:
    """
    Uses the M2M100 model to translate the English input text into all selected languages
    except English itself.
    Each translation is appended as a new InputText entry in state['input_text'].
    """
    print("\n🌍 NODE: translate_to_multiple_node")

    english_texts = [item for item in state["input_text"] if item["language"] == "en"]
    if not english_texts:
        print("❗ No English input text found. Skipping translation.")
        return state

    source_text = english_texts[0]["text"]
    target_languages = [lang for lang in state["selected_languages"] if lang != "en"]
    print(f"   📝 Source text (EN): '{source_text}'")
    print(f"   🌎 Target languages (excluding EN): {', '.join(target_languages)}")

    # --- Load local translation model ---
    model_path = "models/translation/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_path)

    translated_entries: list[InputText] = []

    for lang in target_languages:
        tokenizer.src_lang = "en"

        encoded = tokenizer(source_text, return_tensors="pt")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(lang))
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        translated_entries.append({
            "language": lang,
            "text": translated_text
        })

        print(f"   ✓ Translated to {lang}: {translated_text[:80]}...")

    updated_input_text = state["input_text"] + translated_entries
    return {**state, "input_text": updated_input_text}
