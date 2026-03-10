# src/infer.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import sys
import os

MODEL_DIR = os.path.join("models", "t5-date-normalizer")

def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model dir not found: {MODEL_DIR}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

def normalize_date(text: str) -> str:
    tokenizer, model = load_model()
    prompt = f"normalize date: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=16,
            num_beams=4,
            length_penalty=0.1,
            early_stopping=True,
        )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py \"05Ju1'22\"")
        sys.exit(1)
    text = sys.argv[1]
    result = normalize_date(text)
    print(f"INPUT: {text}")
    print(f"OUTPUT: {result if result else '<empty>'}")