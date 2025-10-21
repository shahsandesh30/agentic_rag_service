# app/corpus/ingest_cli.py
"""
Command-line interface for document ingestion.

This module provides a CLI for ingesting documents into the database
with comprehensive error handling and progress reporting.
"""

import argparse
import logging
import os
import sys

from .ingest import ingest_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Main CLI function for document ingestion.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Ingest PDFs/TXT/MD files into SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.corpus.ingest_cli /path/to/documents
  python -m app.corpus.ingest_cli document.pdf --db my_database.db --source policies
  python -m app.corpus.ingest_cli /data --max-chars 2000 --overlap 200
        """,
    )

    parser.add_argument("path", help="File or folder path to ingest")
    parser.add_argument("--db", default="rag_local.db", help="SQLite database file path")
    parser.add_argument(
        "--source", default="local", help="Logical source label (e.g., 'policies', 'kb')"
    )
    parser.add_argument("--max-chars", type=int, default=1200, help="Maximum characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Character overlap between chunks")
    parser.add_argument(
        "--no-heading-aware", action="store_true", help="Disable heading-aware segmentation"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually ingesting",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate arguments
    if not os.path.exists(args.path):
        logger.error(f"Path does not exist: {args.path}")
        return 1

    if args.max_chars <= 0:
        logger.error("max-chars must be positive")
        return 1

    if args.overlap < 0:
        logger.error("overlap must be non-negative")
        return 1

    if args.overlap >= args.max_chars:
        logger.warning(
            f"Overlap ({args.overlap}) >= max-chars ({args.max_chars}), reducing overlap"
        )
        args.overlap = max(0, args.max_chars - 1)

    try:
        if args.dry_run:
            logger.info("DRY RUN MODE - No actual ingestion will occur")
            # TODO: Implement dry-run functionality
            logger.info(f"Would process: {args.path}")
            logger.info(
                f"Parameters: max_chars={args.max_chars}, overlap={args.overlap}, heading_aware={not args.no_heading_aware}"
            )
            return 0

        logger.info(f"Starting ingestion from: {args.path}")
        logger.info(f"Database: {args.db}")
        logger.info(f"Source: {args.source}")
        logger.info(
            f"Chunking: max_chars={args.max_chars}, overlap={args.overlap}, heading_aware={not args.no_heading_aware}"
        )

        # Perform ingestion
        stats = ingest_path(
            args.path,
            db_path=args.db,
            source=args.source,
            max_chars=args.max_chars,
            overlap=args.overlap,
            heading_aware=(not args.no_heading_aware),
        )

        # Report results
        print("\n=== Ingestion Complete ===")
        print(f"Documents processed: {stats['documents']}")
        print(f"Chunks created: {stats['chunks']}")
        print(f"Database: {stats['db']}")

        if stats.get("processed_files"):
            print(f"Processed files: {len(stats['processed_files'])}")
            if args.verbose:
                for file_path in stats["processed_files"]:
                    print(f"  ✓ {file_path}")

        if stats.get("skipped_files"):
            print(f"Skipped files: {len(stats['skipped_files'])}")
            if args.verbose:
                for file_path in stats["skipped_files"]:
                    print(f"  ✗ {file_path}")

        logger.info("Ingestion completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
