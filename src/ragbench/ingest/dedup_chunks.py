#!/usr/bin/env python3
"""
Chunk deduplication utility.
Removes chunks with identical content_sha256 while preserving reading order.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def unique_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate chunks by content_sha256, preserve first occurrence."""
    seen = set()
    out = []
    
    for chunk in chunks:
        key = chunk.get("content_sha256", "")
        if not key:
            # No hash means we can't dedup safely, include it
            out.append(chunk)
            continue
            
        if key in seen:
            continue
            
        # Keep figures only if they have meaningful text or are images with paths
        if chunk["role"] == "figure":
            if len(chunk["text"].strip()) == 0 and not chunk.get("image_path"):
                continue  # Skip empty figures with no image
                
        seen.add(key)
        out.append(chunk)
    
    return out

def update_order_ranks(chunks: List[Dict[str, Any]]) -> None:
    """Update order_rank to maintain sequence after deduplication."""
    for idx, chunk in enumerate(chunks):
        chunk["order_rank"] = idx

def update_prev_next_ids(chunks: List[Dict[str, Any]]) -> None:
    """Update prev_id/next_id links after deduplication."""
    for i, chunk in enumerate(chunks):
        chunk["prev_id"] = chunks[i-1]["chunk_id"] if i > 0 else None
        chunk["next_id"] = chunks[i+1]["chunk_id"] if i < len(chunks) - 1 else None

def main():
    parser = argparse.ArgumentParser(description="Deduplicate chunks by content hash")
    parser.add_argument("input", type=Path, help="Input chunks JSONL file")
    parser.add_argument("output", type=Path, help="Output deduplicated chunks JSONL file")
    parser.add_argument("--stats", action="store_true", help="Print deduplication stats")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    # Load chunks
    chunks = []
    with args.input.open() as f:
        for line in f:
            chunks.append(json.loads(line))
    
    original_count = len(chunks)
    
    # Deduplicate
    unique = unique_chunks(chunks)
    
    # Update order and links
    update_order_ranks(unique)
    update_prev_next_ids(unique)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for chunk in unique:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    if args.stats:
        removed_count = original_count - len(unique)
        print(f"Original chunks: {original_count}")
        print(f"Unique chunks: {len(unique)}")
        print(f"Removed duplicates: {removed_count}")
        print(f"Deduplication rate: {removed_count/original_count*100:.1f}%")
    
    return 0

if __name__ == "__main__":
    exit(main())
