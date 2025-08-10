from transformers import pipeline as pl
import torch

model_id = "openai/gpt-oss-20b"
pipe = pl(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto"
)

message = [
    {"role":"user", "context":"Explain Masked Language Modeling clearly and concisely."}
]

output = pipe(
    message,
    max_new_tokens=256
)

print(output[0]["generated_text"][-1])
