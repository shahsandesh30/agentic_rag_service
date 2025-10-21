# app/scripts/ingest_and_index.py
import json
import os

import faiss
import numpy as np
from tqdm import tqdm

from app.corpus.ingest import ingest_path
from app.corpus.schema import (
    assign_faiss_ids,
    connect,
    fetch_full_chunks,
    init_db,
    missing_embedding_chunk_ids,
    upsert_embeddings,
)
from app.embed.model import Embedder

FAISS_PATH = "faiss_index/index.faiss"
DB_PATH = "rag_local.db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def ensure_faiss_index(dim: int, metric: str = "cosine") -> faiss.Index:
    """
    Create or load a FAISS index with ID support.
    """
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    if os.path.exists(FAISS_PATH):
        print(f"[INFO] Loading existing FAISS index: {FAISS_PATH}")
        return faiss.read_index(FAISS_PATH)

    print(f"[INFO] Creating new FAISS index (dim={dim}, metric={metric})")
    if metric == "cosine":
        base_index = faiss.IndexFlatIP(dim)
    else:
        base_index = faiss.IndexFlatL2(dim)

    # Wrap in ID map so add_with_ids works
    return faiss.IndexIDMap(base_index)


def main():
    # --- Step 1. Ingest corpus into SQLite ---
    print("[STEP 1] Ingesting documents...")
    stats = ingest_path("data", db_path=DB_PATH)
    print(
        f"[INFO] Ingested {stats['documents']} documents, {stats['chunks']} chunks into {DB_PATH}"
    )

    # --- Step 2. Init embedder ---
    print("[STEP 2] Loading embedder...")
    embedder = Embedder(model_name=MODEL_NAME)
    dim = embedder.dim
    print(f"[INFO] Embedder dimension = {dim}")

    # --- Step 3. Init FAISS index ---
    print("[STEP 3] Preparing FAISS index...")
    index = ensure_faiss_index(dim)

    # --- Step 4. Find chunks missing embeddings ---
    conn = connect(DB_PATH)
    init_db(conn)
    todo = missing_embedding_chunk_ids(conn)
    print(f"[INFO] New chunks to embed: {len(todo)}")

    if not todo:
        print("[DONE] No new chunks found.")
        return

    # --- Step 5. Embed + insert into FAISS ---
    batch_size = 64
    all_ids, all_vecs = [], []

    for i in tqdm(range(0, len(todo), batch_size), desc="Embedding chunks"):
        batch = todo[i : i + batch_size]
        chunk_ids = [cid for cid, _ in batch]
        texts = fetch_full_chunks(conn, chunk_ids)
        vecs = embedder.encode([texts[cid] for cid in chunk_ids])
        all_ids.extend(chunk_ids)
        all_vecs.append(vecs)

    all_vecs = np.vstack(all_vecs).astype("float32")

    # Normalize if cosine similarity
    faiss.normalize_L2(all_vecs)

    # Assign FAISS IDs sequentially
    start_id = index.ntotal
    faiss_ids = list(range(start_id, start_id + len(all_ids)))

    # --- Step 6. Add to FAISS ---
    print(f"[INFO] Adding {len(all_ids)} vectors to FAISS...")
    index.add_with_ids(all_vecs, np.array(faiss_ids, dtype="int64"))

    # --- Step 7. Save FAISS index ---
    faiss.write_index(index, FAISS_PATH)
    meta = {"model": MODEL_NAME, "dim": dim, "index_type": "IndexIDMap(FlatIP)", "metric": "cosine"}
    with open("faiss_index/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved FAISS index at {FAISS_PATH}")

    # --- Step 8. Record mapping in SQLite ---
    assign_faiss_ids(
        conn, [(cid, fid, dim, MODEL_NAME) for cid, fid in zip(all_ids, faiss_ids, strict=False)]
    )
    upsert_embeddings(
        conn,
        [
            (cid, MODEL_NAME, dim, vec.tobytes(), fid)
            for cid, vec, fid in zip(all_ids, all_vecs, faiss_ids, strict=False)
        ],
    )

    with open("faiss_index/ids.json", "w", encoding="utf-8") as f:
        json.dump({cid: fid for cid, fid, *_ in zip(all_ids, faiss_ids, strict=False)}, f, indent=2)
    conn.commit()
    conn.close()

    print("[DONE] Ingestion + indexing completed.")


if __name__ == "__main__":
    main()
