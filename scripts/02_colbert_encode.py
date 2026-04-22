import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# loading ColBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to("cpu")

# parsing passages from .txt file output from 01_chunk_txt.py
passages = []
with open(BASE_DIR/"data"/"passages"/"samplepdf.passages.txt", "r", encoding="utf-8") as f:
    for line in f:
        passages.append(line.strip())

CLS_ID = 101
SEP_ID = 102
# setting batch size for encoding
batch_size = 3
all_embeddings = []
token_counts = []

for start in range(0, len(passages), batch_size):
    end = start + batch_size
    batch = passages[start:end]
    tokens = tokenizer(batch, padding=True, truncation=True, max_length=300, return_tensors="pt")
    tokens = {k: v.to("cpu") for k, v in tokens.items()}
    outputs = model(input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    token_type_ids=tokens["token_type_ids"])
    embeddings = outputs.last_hidden_state

    for i in range(len(batch)):
        input_ids = tokens["input_ids"][i]
        attention_mask = tokens["attention_mask"][i]
        keep = (attention_mask == 1) & (input_ids != CLS_ID) & (input_ids != SEP_ID)
        
        doc_emb = embeddings[i][keep]
        doc_emb = doc_emb.detach().cpu().numpy().astype(np.float32)
        
        all_embeddings.append(doc_emb)
        token_counts.append(int(doc_emb.shape[0]))

big = np.concatenate(all_embeddings, axis=0)
np.save(BASE_DIR/"colbert_out"/"bow"/"doc_embeddings.npy", big)
with open(BASE_DIR/"colbert_out"/"bow"/"token_counts.json", "w", encoding="utf-8") as f:
    json.dump(token_counts, f)

print("Saved embeddings:", big.shape)
print("Saved token counts:", len(token_counts), "docs,", sum(token_counts), "tokens total")



