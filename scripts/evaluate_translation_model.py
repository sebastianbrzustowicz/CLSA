from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODEL_PATH = "facebook/m2m100_418M"

tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH).to("cuda")

languages = [
    "en",  # English
    "de",  # German
    "es",  # Spanish
    "fr",  # French
    "pl",  # Polish
    "ru",  # Russian
    "zh",  # Chinese
    "ja",  # Japanese
    "ko",  # Korean
    "ar",  # Arabic
    "hi",   # Hindi
    "en"
]

texts = [
    "Global cooperation is essential to fight climate change.",
    "Artificial intelligence will reshape the future of work."
]

for text in texts:
    print(f"\nOriginal (EN): {text}")
    current_text = text

    for i in range(len(languages) - 1):
        src_lang = languages[i]
        tgt_lang = languages[i + 1]

        tokenizer.src_lang = src_lang
        encoded = tokenizer(current_text, return_tensors="pt").to("cuda")

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=200
        )

        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"{src_lang.upper()} → {tgt_lang.upper()}: {translated}")
        current_text = translated
