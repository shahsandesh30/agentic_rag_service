import argparse, os, sys
from .ingest import ingest_path

def main():
    p = argparse.ArgumentParser(description="Ingest PDFs/TXT/MD into SQLite (documents & chunks).")
    p.add_argument("path", help="File or folder path to ingest")
    p.add_argument("--db", default="rag_local.db", help="SQLite file (default: rag_local.db)")
    p.add_argument("--source", default="local", help="Logical source label (e.g., 'policies', 'kb')")
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--no-heading-aware", action="store_true", help="Disable heading-aware segmentation")
    args = p.parse_args()

    stats = ingest_path(
        args.path,
        db_path=args.db,
        source=args.source,
        max_chars=args.max_chars,
        overlap=args.overlap,
        heading_aware=(not args.no_heading_aware)
    )
    print(f"Ingested: {stats['documents']} documents, {stats['chunks']} chunks -> DB: {stats['db']}")

if __name__ == "__main__":
    main()
