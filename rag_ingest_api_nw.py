#!/usr/bin/env python3

import requests
import json
import numpy as np
import faiss
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse

# =========================
# CONFIG
# =========================
URLS = [
    "https://en.wikipedia.org/wiki/Netherlands",
    "https://en.wikipedia.org/wiki/France",
    "https://en.wikipedia.org/wiki/Germany",
    "https://public-dns.info/",
]

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 180

FAISS_PATH = "wiki.index"
META_PATH = "wiki_meta.jsonl"

WIKI_API = "https://en.wikipedia.org/w/api.php"

WIKI_HEADERS = {
    "User-Agent": (
        "RAG-Local-Test/1.0 (contact: anoop.saurav@example.com)"
    )
}

# ============================================================
# Fetch Wikipedia article (plain text)
# ============================================================
def fetch_wiki_text(title):
    if not title:
        return ""

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1
    }

    r = requests.get(
        WIKI_API,
        params=params,
        headers=WIKI_HEADERS,
        timeout=20
    )
    r.raise_for_status()

    data = r.json()

    if "query" not in data or "pages" not in data["query"]:
        return ""

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    return page.get("extract", "")

# ============================================================
# Fetch text from ANY HTML website
# ============================================================
def fetch_html_text(url):
    print(f"   → Fetching HTML content...")
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove JS, CSS, non-content blocks
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n+", "\n", text)  # clean blank lines

    return text.strip()

# ============================================================
# Sentence-based chunking
# ============================================================
def sentence_chunk(text, max_chars=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ============================================================
# MAIN
# ============================================================
def main():
    print("🔹 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    metadata = []

    for url in URLS:
        print(f"\n🔹 Fetching: {url}")
        parsed = urlparse(url)
        title = parsed.path.split("/")[-1]

        # Route based on URL type
        if "wikipedia.org" in parsed.netloc:
            print("   → Detected Wikipedia article")
            text = fetch_wiki_text(title)
        else:
            print("   → Detected generic website")
            text = fetch_html_text(url)

        if not text.strip():
            print(f"❌ No text extracted: {url}")
            continue

        # Chunk text
        chunks = sentence_chunk(text, MAX_CHUNK_CHARS)
        print(f"   → Chunks created: {len(chunks)}")

        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "id": f"{title}_chunk_{chunk_id}",
                "source": url,
                "text": chunk
            })

    if not all_chunks:
        raise RuntimeError("❌ No chunks created. Aborting ingestion.")

    print(f"\n✅ Total chunks created: {len(all_chunks)}")

    # Embeddings
    embeddings = model.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Normalize vectors (cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save FAISS index
    faiss.write_index(index, FAISS_PATH)

    # Save metadata
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("\n🎉 INGESTION COMPLETE")
    print(f"📦 FAISS index : {FAISS_PATH}")
    print(f"📄 Metadata    : {META_PATH}")

if __name__ == "__main__":
    main()
