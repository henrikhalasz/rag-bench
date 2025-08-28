#!/usr/bin/env python3
"""
Quality control checker for chunked documents.
Validates schema compliance, coordinate ranges, and reading order.
"""
import json
import sys
import math
from pathlib import Path

def die(msg):
    print(f"QC FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        die("Usage: python qc_check.py <chunks.jsonl>")
    
    path = Path(sys.argv[1])
    if not path.exists():
        die(f"File not found: {path}")
    
    seen_ids = set()
    prev_rank = -1
    prev_page = None
    chunk_count = 0
    
    with path.open() as f:
        for i, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"Invalid JSON at line {i}: {e}")
            
            chunk_count += 1
            
            # Check required fields
            required_fields = ["chunk_id", "doc_id", "source_file", "page_start", "role", "text", "order_rank"]
            for field in required_fields:
                if field not in r:
                    die(f"Missing required field '{field}' at line {i}")
            
            cid = r["chunk_id"]
            if cid in seen_ids:
                die(f"Duplicate chunk_id at line {i}: {cid}")
            seen_ids.add(cid)
            
            # Check bbox validity if present
            b = r.get("bbox_union_norm")
            if b:
                for k in ("x1", "y1", "x2", "y2"):
                    if k not in b:
                        die(f"Missing bbox coordinate '{k}' in {cid}")
                    v = float(b[k])
                    if not (0.0 <= v <= 1.0):
                        die(f"Bbox coordinate {k}={v} out of [0,1] range in {cid}")
                
                if not (b["x2"] >= b["x1"] and b["y2"] >= b["y1"]):
                    die(f"Invalid bbox geometry in {cid}: x2 < x1 or y2 < y1")
            
            # Check coordinate system
            if r.get("coord_system") != "top-left":
                die(f"coord_system mismatch in {cid}, expected 'top-left', got '{r.get('coord_system')}'")
            
            # Check reading order (monotonic within page, can reset across pages)
            page = r["page_start"]
            rank = r["order_rank"]
            
            if prev_page is None or page != prev_page:
                prev_rank = -1  # Reset for new page
            
            if rank < prev_rank:
                die(f"order_rank regression at {cid}: {rank} < {prev_rank}")
            
            prev_rank = rank
            prev_page = page
            
            # Check text content for non-figure chunks
            if r["role"] != "figure" and len(r["text"].strip()) == 0:
                die(f"Empty text in non-figure chunk {cid}")
            
            # Validate role
            valid_roles = {"title", "h1", "h2", "h3", "paragraph", "list_item", "caption", "table", "figure", "other"}
            if r["role"] not in valid_roles:
                die(f"Invalid role '{r['role']}' in {cid}")
    
    print(f"QC OK: {chunk_count} chunks validated")

if __name__ == "__main__":
    main()
