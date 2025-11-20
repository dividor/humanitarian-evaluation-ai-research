#!/usr/bin/env python3
"""
Complete data reset script for humanitarian evaluation AI research project.

This script performs a comprehensive reset of all data:
1. Stops Docker containers
2. Removes Docker volumes (Qdrant database)
3. Clears all local data directories and files
4. Restarts Docker containers with fresh state

Usage:
    python reset_data.py [--yes]

Arguments:
    --yes    Skip confirmation prompt and proceed with reset
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"→ {description}...")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=Path(__file__).parent
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reset all data for humanitarian evaluation AI research project"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and proceed with reset",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HUMANITARIAN EVALUATION AI - COMPLETE DATA RESET")
    print("=" * 70)
    print()
    print("This will:")
    print("  • Stop all Docker containers")
    print("  • Remove Docker volumes (Qdrant database)")
    print("  • Delete ./data/db/ (Qdrant storage)")
    print("  • Delete ./data/pdfs/")
    print("  • Delete ./data/parsed/")
    print("  • Delete ./data/summaries/")
    print("  • Delete ./data/pdf_metadata.xlsx")
    print("  • Delete ./data/cache/")
    print("  • Restart Docker containers with fresh state")
    print()

    if not args.yes:
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Reset cancelled.")
            sys.exit(0)

    print()
    print("=" * 70)
    print("STARTING DATA RESET")
    print("=" * 70)
    print()

    # Step 1: Stop Docker containers and remove volumes
    print("Step 1: Stopping Docker containers and removing volumes...")
    if not run_command(
        ["docker", "compose", "down", "-v"],
        "Stopping containers and removing volumes",
    ):
        print("✗ Failed to stop containers")
        sys.exit(1)
    print("✓ Containers stopped and volumes removed")
    print()

    # Step 2: Clear data directories
    print("Step 2: Clearing data directories...")

    base_path = Path(__file__).parent / "data"
    dirs_to_clear = [
        base_path / "pdfs",
        base_path / "parsed",
        base_path / "summaries",
        base_path / "cache",
        base_path / "db",  # Qdrant database storage
    ]

    files_to_clear = [base_path / "pdf_metadata.xlsx"]

    for directory in dirs_to_clear:
        if directory.exists():
            print(f"  → Removing {directory.relative_to(Path.cwd())}...")
            shutil.rmtree(directory)
            print(f"  ✓ Removed {directory.relative_to(Path.cwd())}")

    for file_path in files_to_clear:
        if file_path.exists():
            print(f"  → Removing {file_path.relative_to(Path.cwd())}...")
            file_path.unlink()
            print(f"  ✓ Removed {file_path.relative_to(Path.cwd())}")

    print("✓ All data directories and files cleared")
    print()

    # Step 3: Restart Docker containers
    print("Step 3: Starting fresh Docker containers...")
    if not run_command(["docker", "compose", "up", "-d"], "Starting containers"):
        print("✗ Failed to start containers")
        sys.exit(1)

    # Wait for containers to be ready
    print("  → Waiting for containers to be ready...")
    subprocess.run(["sleep", "5"], check=True)

    print("✓ Containers started")
    print()

    # Step 4: Verify status
    print("Step 4: Verifying container status...")
    run_command(["docker", "compose", "ps"], "Checking container status")
    print()

    print("=" * 70)
    print("✓ DATA RESET COMPLETE")
    print("=" * 70)
    print()
    print("All data has been cleared and containers are running fresh.")
    print(
        "You can now run the pipeline with: "
        "docker compose exec pipeline python pipeline.py --max-downloads 2"
    )
    print()


if __name__ == "__main__":
    main()
