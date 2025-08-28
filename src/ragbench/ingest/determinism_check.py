#!/usr/bin/env python3
"""
Determinism checker for chunking pipeline.
Runs chunking twice and compares outputs to detect nondeterministic behavior.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys

def compute_content_hash(chunks_file: Path) -> str:
    """Compute hash of content-relevant fields, excluding timestamps."""
    hasher = hashlib.sha256()
    
    with chunks_file.open() as f:
        for line in f:
            chunk = json.loads(line)
            # Include only deterministic fields
            content_fields = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "role": chunk["role"],
                "text": chunk["text"],
                "html": chunk["html"],
                "image_path": chunk["image_path"],
                "bbox_union_norm": chunk.get("bbox_union_norm"),
                "source_element_ids": chunk["source_element_ids"],
                "prev_id": chunk.get("prev_id"),
                "next_id": chunk.get("next_id"),
                "order_rank": chunk["order_rank"],
                "content_sha256": chunk["content_sha256"]
            }
            
            # Sort keys for deterministic serialization
            content_str = json.dumps(content_fields, sort_keys=True, ensure_ascii=False)
            hasher.update(content_str.encode('utf-8'))
    
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Check chunking determinism")
    parser.add_argument("--input", type=Path, required=True, help="Parser JSONL file")
    parser.add_argument("--chunker", type=Path, 
                       default=Path("src/ragbench/ingest/chunker.py"),
                       help="Path to chunker script")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--overlap-tokens", type=int, default=120)
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    if not args.chunker.exists():
        print(f"ERROR: Chunker script not found: {args.chunker}", file=sys.stderr)
        return 1
    
    # Create temporary directories for two runs
    with tempfile.TemporaryDirectory(prefix="chunk_det_1_") as tmpdir1, \
         tempfile.TemporaryDirectory(prefix="chunk_det_2_") as tmpdir2:
        
        tmppath1 = Path(tmpdir1)
        tmppath2 = Path(tmpdir2)
        
        # Run chunker twice with identical parameters
        base_cmd = [
            sys.executable, "-m", "src.ragbench.ingest.chunker",
            "--input", str(args.input),
            "--max-tokens", str(args.max_tokens),
            "--overlap-tokens", str(args.overlap_tokens)
        ]
        
        print("Running chunker (attempt 1)...")
        result1 = subprocess.run(
            base_cmd + ["--out", str(tmppath1)],
            capture_output=True, text=True
        )
        
        if result1.returncode != 0:
            print(f"ERROR: First chunker run failed:\n{result1.stderr}", file=sys.stderr)
            return 1
        
        print("Running chunker (attempt 2)...")
        result2 = subprocess.run(
            base_cmd + ["--out", str(tmppath2)],
            capture_output=True, text=True
        )
        
        if result2.returncode != 0:
            print(f"ERROR: Second chunker run failed:\n{result2.stderr}", file=sys.stderr)
            return 1
        
        # Find the chunks files
        chunks_files1 = list(tmppath1.glob("*_chunks.jsonl"))
        chunks_files2 = list(tmppath2.glob("*_chunks.jsonl"))
        
        if len(chunks_files1) != 1 or len(chunks_files2) != 1:
            print("ERROR: Expected exactly one chunks file per run", file=sys.stderr)
            return 1
        
        chunks1 = chunks_files1[0]
        chunks2 = chunks_files2[0]
        
        # Compare content hashes
        hash1 = compute_content_hash(chunks1)
        hash2 = compute_content_hash(chunks2)
        
        print(f"Run 1 content hash: {hash1}")
        print(f"Run 2 content hash: {hash2}")
        
        if hash1 == hash2:
            print("✅ DETERMINISM OK: Identical outputs")
            return 0
        else:
            print("❌ DETERMINISM FAIL: Outputs differ")
            print(f"Files for inspection:")
            print(f"  Run 1: {chunks1}")
            print(f"  Run 2: {chunks2}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
