# app/scripts/add_dataset_faiss.py
import argparse
import json
import os
import pathlib

import faiss
import numpy as np

from app.corpus.chunk import chunk_file
from app.embed.model import Embedder

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.json")


def load_index(dim: int):
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        # Wrap if not already wrapped
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
    else:
        base_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIDMap(base_index)
    return index


def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index(index, meta):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Add dataset(s) directly into FAISS index (no SQLite)."
    )
    ap.add_argument("paths", nargs="+", help="Files or directories to ingest")
    args = ap.parse_args()

    embedder = Embedder(model_name=MODEL_NAME)
    print(f"Using embedding model '{MODEL_NAME}' with dim={embedder.dim}")
    meta = load_meta()
    print(f"Loaded {len(meta)} existing chunks from metadata.")
    index = load_index(embedder.dim)
    print(f"FAISS index has {index} vectors.")

    next_id = max(map(int, meta.keys()), default=0) + 1

    files = []
    for p in args.paths:
        path = pathlib.Path(p)
        if path.is_dir():
            files.extend(path.rglob("*.txt"))
            files.extend(path.rglob("*.md"))
            files.extend(path.rglob("*.pdf"))
        elif path.is_file():
            files.append(path)

    total_chunks = 0
    for f in files:
        docs = chunk_file(str(f))
        texts = [d["text"] for d in docs]
        if not texts:
            continue
        vecs = embedder.encode(texts)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # normalize

        start_id = next_id
        ids = list(range(start_id, start_id + len(docs)))
        index.add_with_ids(vecs.astype("float32"), np.array(ids))

        for i, d in zip(ids, docs, strict=False):
            meta[i] = {
                "text": d["text"],
                "path": d["path"],
                "section": d["section"],
                "source": d["source"],
            }

        next_id += len(docs)
        total_chunks += len(docs)
        print(f"Ingested {len(docs)} chunks from {f}")

    save_index(index, meta)
    print(f"✅ Added {total_chunks} new chunks. FAISS index updated at {INDEX_FILE}")


if __name__ == "__main__":
    main()
