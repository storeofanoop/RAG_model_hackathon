import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Wiki RAG Demo", page_icon="🧠")

import faiss
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# --- Config ---
FAISS_PATH = "wiki.index"
META_PATH = "wiki_meta.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 3

# --- Load index & metadata ---
@st.cache_resource
def load_resources():
    index = faiss.read_index(FAISS_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = [json.loads(line) for line in f]

    model = SentenceTransformer(EMBED_MODEL)
    return index, metas, model

index, metas, model = load_resources()

def clean_chunk(text, max_len=400):
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text

# --- UI ---
st.title("🧠 RAG-pedia (your friend)")

TOP_K = 3

query = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is the capital of France?"
)

if query:
    with st.spinner("Searching knowledge base..."):
        q_emb = model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        scores, indices = index.search(q_emb.astype(np.float32), TOP_K)
        top_chunks = [metas[i] for i in indices[0]]

        context = "\n\n".join(c["text"] for c in top_chunks)

        prompt = f"""
You are a helpful assistant answering questions using the provided context.

Rules:
- Answer in ONE short sentence.
- If the question asks for a specific fact (e.g., capital, population, date),
  extract ONLY that fact.
- Do NOT provide background or explanation unless explicitly asked.
- Do NOT quote the context verbatim.
- If the answer is not clearly present, say:
  "Information not found in the provided documents."

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer:
"""

    st.subheader("📘 Retrieved Context")
    for c in top_chunks:
        st.markdown(f"**Source:** {c['source']}")
        st.write(c["text"][:250] + "...")

    if st.button("Ask Ollama"):
        with st.spinner("Talking to Ollama..."):
            payload = {
                "model": "llama3:70b",  # or phi3
                "prompt": prompt,
                "stream": False
            }
            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
                answer = resp.json().get("response", "No response")
            except Exception as e:
                answer = f"❌ Ollama error: {e}"

        answer = answer.strip()
        answer = answer.split("\n")[0]   # keep first paragraph
        if len(answer) > 400:
            answer = answer[:400].rsplit(" ", 1)[0] + "..."

        st.subheader("💬 Answer")
        st.write(answer)
