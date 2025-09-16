# app/scripts/build_index.py
import argparse, json, os
import numpy as np
from app.retrieval.store import connect
from app.vector_store.faiss_store import FaissStore
from app.vector_store.qdrant_store import QdrantStore

def load_embeddings(conn, model: str):
    rows = conn.execute("SELECT chunk_id, dim, vec FROM embeddings WHERE model = ?", (model,)).fetchall()
    ids, vecs = [], []
    dim = 0
    for r in rows:
        ids.append(r["chunk_id"])
        dim = r["dim"]
        v = np.frombuffer(r["vec"], dtype=np.float32, count=dim)
        vecs.append(v)
    if not vecs:
        return [], np.zeros((0,1), dtype=np.float32), 0
    mat = np.vstack(vecs).astype(np.float32, copy=False)
    # cosine: store normalized
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norm
    return ids, mat, dim

def load_payloads(conn, ids):
    payloads = []
    if not ids:
        return payloads
    # join basic meta (optional)
    for i in range(0, len(ids), 999):
        batch = ids[i:i+999]
        q = f"SELECT id, section, meta_json FROM chunks WHERE id IN ({','.join('?'*len(batch))})"
        meta = {r["id"]: r for r in conn.execute(q, batch)}
        for cid in batch:
            r = meta.get(cid)
            if r:
                payloads.append({"chunk_id": cid, "section": r["section"], "meta_json": r["meta_json"]})
            else:
                payloads.append({"chunk_id": cid})
    return payloads

def main():
    ap = argparse.ArgumentParser(description="Build vector index from SQLite embeddings.")
    ap.add_argument("--db", default="rag_local.db")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--store", choices=["faiss","qdrant"], required=True)
    args = ap.parse_args()

    conn = connect(args.db)
    ids, mat, dim = load_embeddings(conn, args.model)
    payloads = load_payloads(conn, ids)
    conn.close()
    if len(ids)==0:
        print("No embeddings found for model", args.model); return

    if args.store == "faiss":
        store = FaissStore(dim=dim, path="faiss_index", metric="cosine")
    else:
        store = QdrantStore(dim=dim, collection="chunks")

    store.build(ids, mat, payloads)
    print(f"Built {args.store} index with {len(ids)} vectors (dim={dim})")

if __name__ == "__main__":
    main()
