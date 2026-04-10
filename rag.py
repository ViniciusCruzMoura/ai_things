# rag_qwen_example.py
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1) Documents (small knowledge base)
docs = [
    "The Eiffel Tower is in Paris and was completed in 1889.",
    "Python is a programming language commonly used for machine learning.",
    "The mitochondrion is the powerhouse of the cell.",
    "Hugging Face provides tools for transformers and model hosting.",
]

# 2) Build embeddings for the docs (use a small SBERT model)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight
doc_embeddings = embed_model.encode(docs, convert_to_numpy=True)

# 3) Create a FAISS index
d = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings)  # add vectors

# helper to retrieve top-k docs
def retrieve(query: str, k: int = 2) -> List[str]:
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, k)
    idxs = idxs[0]
    return [docs[i] for i in idxs]

# 4) Load generator model (qwen3-0.6B)
# NOTE: If qwen3-0.6B is hosted or requires special loading, change below accordingly.
model_name = "Qwen/Qwen3-0.6B"  # placeholder — replace with actual HF repo ID if available
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 5) RAG generation: retrieve, build prompt, and generate
def rag_answer(query: str, top_k: int = 2, max_new_tokens: int = 200) -> str:
    retrieved = retrieve(query, k=top_k)
    # Build a simple prompt that conditions the generator on retrieved context
    context = "\n\n".join(f"Context {i+1}: {r}" for i, r in enumerate(retrieved))
    prompt = (
        "You are a helpful assistant. Use the following retrieved context to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

# 6) Example usage
if __name__ == "__main__":
    q = "Where is the Eiffel Tower and when was it completed?"
    answer = rag_answer(q)
    print("Q:", q)
    print("A:", answer)

