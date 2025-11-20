#!/usr/bin/env python3
"""Debug script to check document status in Qdrant"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.db import db

docs = db.client.scroll(collection_name="documents", limit=10)[0]
print(f"Total docs: {len(docs)}\n")

for i, d in enumerate(docs[:5], 1):
    print(f"Document {i}:")
    print(f"  ID: {d.id}")
    if d.payload:
        print(f"  Title: {d.payload.get('title', 'N/A')[:60]}")
        print(f"  Status: {d.payload.get('status', 'N/A')}")
        print(f"  Filepath: {d.payload.get('filepath', 'N/A')}")
        print(f"  Download error: {d.payload.get('download_error', 'None')[:100]}")
    else:
        print("  Payload: None")
    print()
