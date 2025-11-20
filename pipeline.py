#!/usr/bin/env python3
"""
Complete pipeline orchestrator for humanitarian evaluation document processing.

Runs the full pipeline in order:
1. Download PDFs from UN Evaluation repository
2. Parse PDFs to JSON/markdown
3. Summarize documents
4. Index documents and chunks into Qdrant
5. Display statistics and status

Usage:
    python pipeline.py [--max-downloads N] [--skip-download]
    [--skip-parse] [--skip-summarize] [--skip-index]
"""

import argparse
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, step_name: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run as list of strings
        step_name: Human-readable name for logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*60}")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ {step_name} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete humanitarian evaluation document processing pipeline"
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=2,
        help="Maximum number of documents to download (default: 2)",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip the download step"
    )
    parser.add_argument("--skip-parse", action="store_true", help="Skip the parse step")
    parser.add_argument(
        "--skip-summarize", action="store_true", help="Skip the summarize step"
    )
    parser.add_argument("--skip-index", action="store_true", help="Skip the index step")
    parser.add_argument(
        "--limit-index",
        type=int,
        default=999999,
        help="Maximum number of documents to index (default: unlimited)",
    )

    args = parser.parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("HUMANITARIAN EVALUATION DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Max downloads: {args.max_downloads}")
    logger.info(f"Skip download: {args.skip_download}")
    logger.info(f"Skip parse: {args.skip_parse}")
    logger.info(f"Skip summarize: {args.skip_summarize}")
    logger.info(f"Skip index: {args.skip_index}")

    # Track success of each step
    steps_completed = []
    steps_failed = []

    # Step 1: Download
    if not args.skip_download:
        success = run_command(
            [
                "python",
                "pipeline/download.py",
                "--max-results",
                str(args.max_downloads),
            ],
            f"Download (max {args.max_downloads} documents)",
        )
        if success:
            steps_completed.append("Download")
        else:
            steps_failed.append("Download")
            logger.error("Download step failed. Stopping pipeline.")
            sys.exit(1)
    else:
        logger.info("\n⏭️  Skipping download step")

    # Step 2: Parse
    if not args.skip_parse:
        success = run_command(["python", "pipeline/parse.py"], "Parse documents")
        if success:
            steps_completed.append("Parse")
        else:
            steps_failed.append("Parse")
            logger.warning("Parse step failed. Continuing with remaining steps...")
    else:
        logger.info("\n⏭️  Skipping parse step")

    # Step 3: Summarize
    if not args.skip_summarize:
        success = run_command(
            ["python", "pipeline/summarize.py"], "Summarize documents"
        )
        if success:
            steps_completed.append("Summarize")
        else:
            steps_failed.append("Summarize")
            logger.warning("Summarize step failed. Continuing with remaining steps...")
    else:
        logger.info("\n⏭️  Skipping summarize step")

    # Step 4: Index
    if not args.skip_index:
        success = run_command(
            [
                "python",
                "pipeline/index.py",
                "--limit",
                str(args.limit_index),
                "--force",
            ],
            f"Index documents (max {args.limit_index})",
        )
        if success:
            steps_completed.append("Index")
        else:
            steps_failed.append("Index")
            logger.warning("Index step failed. Continuing to status...")
    else:
        logger.info("\n⏭️  Skipping index step")

    # Final: Show status
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE STATUS & STATISTICS")
    logger.info("=" * 60)

    run_command(["python", "scripts/status.py"], "Status Report")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    if steps_completed:
        logger.info(f"✅ Completed steps: {', '.join(steps_completed)}")

    if steps_failed:
        logger.error(f"❌ Failed steps: {', '.join(steps_failed)}")
        sys.exit(1)
    else:
        logger.info("\n🎉 Pipeline completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
