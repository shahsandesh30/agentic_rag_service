# app/embed/compute_cli.py
import argparse
from .compute import compute_embeddings

def main():
    p = argparse.ArgumentParser(description="Compute embeddings for chunks and store in SQLite.")
    p.add_argument("--db", default="rag_local.db", help="SQLite file (default: rag_local.db)")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="HF embedding model id")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit-per-pass", type=int, default=5000)
    args = p.parse_args()

    stats = compute_embeddings(
        db_path=args.db,
        model_name=args.model,
        batch_size=args.batch_size,
        limit_per_pass=args.limit_per_pass,
    )
    print(f"Embedded {stats['embedded']} chunks with {stats['model']} -> {stats['db']}")

if __name__ == "__main__":
    main()
