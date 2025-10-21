# app/agent/agent_cli.py
"""
Command-line interface for the agent system.

This module provides a CLI for running the agent with comprehensive
error handling, logging, and configuration options.
"""

import argparse
import json
import logging
import sys
from typing import Any

from app.agent.graph import run_agent
from app.agent.types import AgentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Main CLI function for running the agent.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run agent over a question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.agent.agent_cli "What is the legal definition of assault?"
  python -m app.agent.agent_cli "Hello there" --trace
  python -m app.agent.agent_cli "Latest news about AI" --verbose
        """,
    )

    parser.add_argument("question", help="User question to process")
    parser.add_argument("--trace", action="store_true", help="Include trace information in output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", help="Path to configuration file (JSON)")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["json", "text"], default="json", help="Output format")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate arguments
    if not args.question or not args.question.strip():
        logger.error("Question cannot be empty")
        return 1

    try:
        logger.info(f"Processing question: {args.question[:100]}...")

        # Load configuration if provided
        config = None
        if args.config:
            try:
                with open(args.config) as f:
                    config_data = json.load(f)
                    config = AgentConfig(**config_data)
                    logger.debug(f"Loaded configuration from {args.config}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                return 1

        # Run the agent
        result = run_agent(args.question)

        # Prepare output
        if args.trace:
            output_data = result
        else:
            output_data = result.get("final", {})

        # Format output
        if args.format == "text":
            output_text = _format_as_text(output_data)
        else:
            output_text = json.dumps(output_data, ensure_ascii=False, indent=2)

        # Write output
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output_text)
                logger.info(f"Output written to {args.output}")
            except Exception as e:
                logger.error(f"Failed to write output file: {e}")
                return 1
        else:
            print(output_text)

        logger.info("Agent processing completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Agent processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _format_as_text(data: dict[str, Any]) -> str:
    """
    Format agent output as human-readable text.

    Args:
        data: Agent output data

    Returns:
        Formatted text string
    """
    if not data:
        return "No response generated."

    # Extract main components
    answer = data.get("answer", "No answer provided.")
    confidence = data.get("confidence", 0.0)
    citations = data.get("citations", [])
    safety = data.get("safety", {})

    # Build formatted output
    lines = []
    lines.append("=" * 60)
    lines.append("AGENT RESPONSE")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Answer: {answer}")
    lines.append("")
    lines.append(f"Confidence: {confidence:.2f}")

    if safety:
        lines.append(f"Safety: {safety.get('level', 'unknown')}")
        if safety.get("reason"):
            lines.append(f"Safety Reason: {safety['reason']}")

    if citations:
        lines.append("")
        lines.append("Citations:")
        for i, citation in enumerate(citations, 1):
            if isinstance(citation, dict):
                source = citation.get("source", "Unknown")
                path = citation.get("path", "")
                section = citation.get("section", "")
                lines.append(f"  {i}. {source}")
                if path:
                    lines.append(f"     Path: {path}")
                if section:
                    lines.append(f"     Section: {section}")
            else:
                lines.append(f"  {i}. {citation}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
