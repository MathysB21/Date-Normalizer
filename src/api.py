# FastAPI service for inference
from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os

MODEL_DIR = os.path.join("models", "t5-date-normalizer")

app = FastAPI(title="T5 Date Normalizer")

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()

class NormalizeRequest(BaseModel):
    input: str

@app.post("/normalize-date/")
def normalize_date(req: NormalizeRequest):
    prompt = f"normalize date: {req.input}"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_length=16,
            num_beams=4,
            length_penalty=0.1,
        )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"normalized_date": result}