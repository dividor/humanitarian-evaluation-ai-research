#!/usr/bin/env python3
"""
Quick status check for the Qdrant pipeline.
Shows document counts by status.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from pipeline.db import db


def main():
    print("\n" + "=" * 60)
    print("QDRANT PIPELINE STATUS")
    print("=" * 60)

    # Get all documents
    docs = list(db.get_all_documents())
    total = len(docs)

    print(f"\nTotal documents: {total}")

    # Count by status
    statuses = Counter([d.get("status", "unknown") for d in docs])

    print("\nStatus breakdown:")
    print("-" * 40)
    for status in [
        "downloaded",
        "parsed",
        "summarized",
        "indexed",
        "download_failed",
        "parse_failed",
        "summarize_failed",
    ]:
        count = statuses.get(status, 0)
        if count > 0:
            print(f"  {status:20s}: {count:3d}")

    # Show other statuses if any
    other_statuses = {
        k: v
        for k, v in statuses.items()
        if k
        not in [
            "downloaded",
            "parsed",
            "summarized",
            "indexed",
            "download_failed",
            "parse_failed",
            "summarize_failed",
        ]
    }
    if other_statuses:
        print("\nOther statuses:")
        for status, count in other_statuses.items():
            print(f"  {status:20s}: {count:3d}")

    # Show recently parsed documents
    parsed_docs = [d for d in docs if d.get("status") == "parsed"]
    if parsed_docs:
        print(f"\nRecently parsed ({len(parsed_docs)} documents):")
        print("-" * 40)
        for doc in parsed_docs[:5]:
            title = doc.get("title", "Unknown")[:50]
            print(f"  • {title}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
