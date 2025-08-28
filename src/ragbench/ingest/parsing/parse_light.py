#!/usr/bin/env python3
"""
PDF parser for RAG benchmarking.
Processes PDFs under data/raw/ and emits normalized JSONL + extracted images.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

from .processors import (
    parse_pdf_with_unstructured, extract_element_data,
    HAS_UNSTRUCTURED, HAS_TESSERACT
)
from .tables import extract_tables_with_camelot, HAS_CAMELOT
from .utils import (
    setup_logging, compute_doc_id, write_jsonl_atomic,
    find_pdf_files
)

logger = logging.getLogger(__name__)


def process_pdf(pdf_path: Path, output_dir: Path, use_hires: bool = True, table_fallback: str = "none") -> Dict[str, Any]:
    """Process a single PDF file and return processing statistics."""
    logger.info(f"Processing {pdf_path.name}...")
    
    doc_id = compute_doc_id(pdf_path)
    output_file = output_dir / f"{pdf_path.stem}.jsonl"
    image_dir = output_dir / f"{pdf_path.stem}_images"
    
    stats = {
        "file": pdf_path.name,
        "success": False,
        "elements": 0,
        "pages": 0,
        "tables": 0,
        "images": 0,
        "strategy": "unknown",
        "error": None
    }
    
    try:
        # Parse PDF
        elements = parse_pdf_with_unstructured(pdf_path, image_dir, use_hires)
        
        # Process elements
        jsonl_records = []
        pages_seen = set()
        has_html_tables = False
        
        for idx, element in enumerate(elements):
            try:
                record = extract_element_data(element, doc_id, pdf_path, image_dir, idx)
                jsonl_records.append(record)
                
                if record["page_number"] is not None:
                    pages_seen.add(record["page_number"])
                
                if record["type"] == "Table":
                    stats["tables"] += 1
                    if record["html"]:
                        has_html_tables = True
                
                if record["type"] == "Image":
                    stats["images"] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process element {idx} in {pdf_path.name}: {e}")
                continue
        
        # Camelot fallback for tables if no HTML tables were found
        if table_fallback == "camelot" and stats["tables"] > 0 and not has_html_tables:
            camelot_records = extract_tables_with_camelot(pdf_path, output_dir, doc_id)
            if camelot_records:
                # Replace table records with Camelot ones
                non_table_records = [r for r in jsonl_records if r["type"] != "Table"]
                jsonl_records = non_table_records + camelot_records
                stats["tables"] = len(camelot_records)
        
        # Post-run sanity warning for missing image paths
        missing = sum(1 for r in jsonl_records if r["type"] == "Image" and not r["image_path"])
        if missing:
            logger.warning(f"{pdf_path.name}: {missing} Image records have empty image_path after rehoming.")
        
        # Write JSONL atomically
        write_jsonl_atomic(output_file, jsonl_records)
        
        # Update stats
        stats.update({
            "success": True,
            "elements": len(jsonl_records),
            "pages": len(pages_seen),
            "strategy": "hi_res" if use_hires else "fast"
        })
        
        logger.info(f"✓ {pdf_path.name}: {stats['elements']} elements, "
                   f"{stats['pages']} pages, {stats['tables']} tables, {stats['images']} images")
        
    except Exception as e:
        error_msg = str(e)
        stats["error"] = error_msg
        logger.error(f"✗ Failed to process {pdf_path.name}: {error_msg}")
    
    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse PDF documents and extract structured data + images"
    )
    parser.add_argument("--src", type=Path, default=Path("data/raw"),
                       help="Source directory containing PDFs (default: data/raw)")
    parser.add_argument("--out", type=Path, default=Path("data/processed"),
                       help="Output directory for processed files (default: data/processed)")
    parser.add_argument("--no-recursive", action="store_true",
                       help="Disable recursive PDF search")
    parser.add_argument("--no-hires", action="store_true",
                       help="Disable hi_res strategy, use fast only")
    parser.add_argument("--table-fallback", choices=["camelot", "none"], default="none",
                       help="Table extraction fallback method (default: none)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (default: 1)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    # Validate dependencies
    if not HAS_UNSTRUCTURED:
        logger.error("unstructured library not installed. Install with: pip install unstructured")
        return 1
    
    # Find PDF files
    if not args.src.exists():
        logger.error(f"Source directory does not exist: {args.src}")
        return 1
    
    pdf_files = find_pdf_files(args.src, recursive=not args.no_recursive)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {args.src}")
        return 1
    
    # Sort files for deterministic processing
    pdf_files.sort()
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Validate Camelot if needed
    if args.table_fallback == "camelot" and not HAS_CAMELOT:
        logger.error("camelot-py not installed but --table-fallback camelot requested. Install with: pip install camelot-py[cv]")
        return 1
    
    # Process files
    use_hires = not args.no_hires
    all_stats = []
    
    for pdf_path in pdf_files:
        stats = process_pdf(pdf_path, args.out, use_hires, args.table_fallback)
        all_stats.append(stats)
    
    # Summary
    successful = [s for s in all_stats if s["success"]]
    failed = [s for s in all_stats if not s["success"]]
    
    total_elements = sum(s["elements"] for s in successful)
    total_pages = sum(s["pages"] for s in successful)
    total_tables = sum(s["tables"] for s in successful)
    total_images = sum(s["images"] for s in successful)
    
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Files processed: {len(successful)}/{len(pdf_files)}")
    logger.info(f"Total elements: {total_elements}")
    logger.info(f"Total pages: {total_pages}")
    logger.info(f"Total tables: {total_tables}")
    logger.info(f"Total images: {total_images}")
    
    if failed:
        logger.warning(f"Failed files: {len(failed)}")
        for stats in failed:
            logger.warning(f"  {stats['file']}: {stats['error']}")
    
    return 0 if successful else 1


if __name__ == "__main__":
    exit(main())
