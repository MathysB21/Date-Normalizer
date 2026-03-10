from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import torch
import os
from tqdm import tqdm

MODEL_DIR = os.path.join("models", "t5-date-normalizer")

def normalize_batch(model, tokenizer, texts):
    prompts = [f"normalize date: {t}" for t in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=16,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).eval()

    df = pd.read_csv("data/test.csv")
    # strip the training prefix if present (defensive)
    inputs = df["input"].tolist()
    inputs = [t.replace("normalize date: ", "") for t in inputs]

    preds = []
    bs = 128
    for i in tqdm(range(0, len(inputs), bs)):
        batch = inputs[i : i + bs]
        out = normalize_batch(model, tokenizer, batch)
        preds.extend(out)

    df["pred"] = preds
    df["ok"] = (df["pred"] == df["target"]).astype(int)
    acc = df["ok"].mean()
    print(f"Test accuracy: {acc:.4f}")
    df.to_csv("data/test_with_preds.csv", index=False)
    print("Wrote data/test_with_preds.csv")