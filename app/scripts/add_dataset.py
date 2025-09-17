# app/scripts/add_dataset.py
import argparse, os, pathlib
from app.retrieval.store import connect
from app.corpus.chunk import chunk_file
from app.embed.model import Embedder

DB_PATH = "rag_local.db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def ingest_file(path: str, conn):
    """Chunk file and insert into SQLite if new."""
    docs = chunk_file(path)  # returns list of dicts with {id, text, section, path, source}
    n_new = 0
    for d in docs:
        try:
            conn.execute(
                "INSERT INTO chunks (id, text, section, path, source, meta_json) VALUES (?,?,?,?,?,?)",
                (d["id"], d["text"], d.get("section"), d.get("path"), d.get("source"), d.get("meta_json")),
            )
            n_new += 1
        except Exception:
            # duplicate (chunk_id already exists)
            pass
    return n_new

def embed_new(conn, embedder):
    """Embed chunks not yet in embeddings table."""
    cur = conn.execute("SELECT id, text FROM chunks")
    rows = cur.fetchall()
    n_emb = 0
    for r in rows:
        cid = r["id"]
        text = r["text"]
        found = conn.execute(
            "SELECT 1 FROM embeddings WHERE model=? AND chunk_id=?",
            (MODEL_NAME, cid)
        ).fetchone()
        if found:
            continue
        vec = embedder.encode([text])[0]
        conn.execute(
            "INSERT INTO embeddings (chunk_id, model, dim, vec) VALUES (?,?,?,?)",
            (cid, MODEL_NAME, len(vec), vec.astype('float32').tobytes())
        )
        n_emb += 1
    return n_emb

def collect_files(paths):
    """Expand files and directories into a flat file list."""
    all_files = []
    exts = [".txt", ".pdf", ".md"]
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            for ext in exts:
                all_files.extend(list(path.rglob(f"*{ext}")))
        elif path.is_file() and path.suffix.lower() in exts:
            all_files.append(path)
    return [str(f) for f in all_files]

def main():
    ap = argparse.ArgumentParser(description="Add new dataset(s) and rebuild FAISS index.")
    ap.add_argument("paths", nargs="+", help="Files or directories to ingest")
    args = ap.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("⚠️ No valid files found to ingest (.txt, .pdf, .md)")
        return

    conn = connect(DB_PATH)
    total_new = 0
    for f in files:
        n = ingest_file(f, conn)
        total_new += n
        print(f"Ingested {n} new chunks from {f}")
    conn.commit()

    embedder = Embedder(model_name=MODEL_NAME)
    n_emb = embed_new(conn, embedder)
    conn.commit()
    conn.close()
    print(f"Embedded {n_emb} new chunks")

    # Rebuild FAISS index
    print("Rebuilding FAISS index...")
    os.system(f"python -m app.scripts.build_index --store faiss --model {MODEL_NAME}")

    print("✅ Dataset added and FAISS index rebuilt")

if __name__ == "__main__":
    main()
