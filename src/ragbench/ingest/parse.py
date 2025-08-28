#!/usr/bin/env python3
"""
PDF parser for RAG benchmarking.
Processes PDFs under data/raw/ and emits normalized JSONL + extracted images.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import warnings

# Suppress unstructured telemetry and warnings
os.environ.setdefault("UNSTRUCTURED_TELEMETRY_DISABLED", "1")
os.environ.setdefault("SCARF_NO_ANALYTICS", "1")
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Element
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False

logger = logging.getLogger(__name__)

# Text normalization constants
_SOFT_HYPHEN = "\u00ad"
_NBSP = "\xa0"


def setup_logging(debug: bool = False) -> None:
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def compute_doc_id(file_path: Path) -> str:
    """Generate stable SHA256 hash of absolute file path for doc_id."""
    return hashlib.sha256(str(file_path.resolve()).encode('utf-8')).hexdigest()[:16]


def normalize_text(text: str) -> str:
    """Normalize whitespace in text while preserving paragraph structure."""
    if not text:
        return ""
    # Unicode and weird whitespace
    try:
        from ftfy import fix_text
        text = fix_text(text, normalization="NFKC")
    except Exception:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "").replace(_NBSP, " ")

    # De-hyphenate at line breaks: "inter-\nnational" -> "international"
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=[a-z])', '', text)

    # Normalize newlines/spaces while preserving paragraphs
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)          # trim around newlines
    paragraphs = [re.sub(r'[ \t]+', ' ', p).strip()
                  for p in re.split(r'\n{2,}', text)]
    paragraphs = [p for p in paragraphs if p]
    return '\n\n'.join(paragraphs)


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract file metadata including last modified time."""
    try:
        stat = file_path.stat()
        return {
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "languages": ["eng"]
        }
    except Exception as e:
        logger.warning(f"Could not get metadata for {file_path}: {e}")
        return {"languages": ["eng"]}


def extract_element_data(element: Element, doc_id: str, source_path: Path, 
                        image_dir: Path, element_idx: int) -> Dict[str, Any]:
    """Extract normalized data from an unstructured element."""
    # Get basic element info
    element_dict = element.to_dict()
    element_type = element_dict.get('type', type(element).__name__)
    meta = getattr(element, 'metadata', None)
    
    # Extract text and normalize
    text = getattr(element, 'text', '') or ''
    normalized_text = normalize_text(text)
    
    # Extract HTML if available (for tables)
    html = ""
    if hasattr(element, 'text_as_html') and element.text_as_html:
        html = element.text_as_html
    elif meta and getattr(meta, 'text_as_html', None):
        html = meta.text_as_html
    
    # ---- Coordinates (authoritative + normalized) ---------------------------
    coord_system = None
    layout_w = None
    layout_h = None
    coordinates = None
    bbox_norm_top_left = None

    coords_obj = None
    if meta and hasattr(meta, "coordinates"):
        coords_obj = meta.coordinates
    elif hasattr(element, "coordinates"):
        coords_obj = element.coordinates

    if coords_obj:
        try:
            # Unstructured objects have .to_dict() with system / layout sizes / points
            cdict = coords_obj.to_dict() if hasattr(coords_obj, "to_dict") else None
            if cdict:
                # points are TL, TR, BR, BL in PixelSpace for hi_res
                pts = cdict.get("points") or []
                if len(pts) >= 4:
                    x1, y1 = float(pts[0][0]), float(pts[0][1])
                    x2, y2 = float(pts[2][0]), float(pts[2][1])
                    if x2 < x1: x1, x2 = x2, x1
                    if y2 < y1: y1, y2 = y2, y1
                    coordinates = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

                coord_system = cdict.get("system")  # e.g., "PixelSpace"
                layout_w = cdict.get("layout_width")
                layout_h = cdict.get("layout_height")

                # normalized top-left if layout size present
                if coordinates and layout_w and layout_h and layout_w > 0 and layout_h > 0:
                    bbox_norm_top_left = {
                        "x1": coordinates["x1"] / float(layout_w),
                        "y1": coordinates["y1"] / float(layout_h),
                        "x2": coordinates["x2"] / float(layout_w),
                        "y2": coordinates["y2"] / float(layout_h),
                    }
        except Exception:
            pass
    
    # Handle images - simplified approach
    image_path = ""
    if meta and getattr(meta, "image_path", None):
        src_path = Path(meta.image_path)
        if src_path.exists():
            try:
                # If Unstructured already used our target dir, just relativize
                image_path = str(src_path.resolve().relative_to(image_dir.parent.resolve()))
            except ValueError:
                # If it saved into a sibling like "figures/", move/copy into <stem>_images/
                dst = image_dir / src_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(str(src_path), str(dst))
                image_path = str(dst.resolve().relative_to(image_dir.parent.resolve()))
    elif hasattr(element, "image_data") and element.image_data:
        # Fallback: save bytes (rare)
        page_num = getattr(meta, "page_number", 0) or 0
        image_path = save_image(element.image_data, image_dir, page_num, element_idx)
    
    # Extract page number
    page_number = None
    if meta and hasattr(meta, 'page_number'):
        page_number = meta.page_number
    
    file_meta = get_file_metadata(source_path)
    # Carry coordinate context for downstream consumers
    if coord_system:
        file_meta["coord_system"] = coord_system  # expected: "PixelSpace" for hi_res
    if layout_w and layout_h:
        file_meta["layout_size"] = {"w": float(layout_w), "h": float(layout_h)}
    if bbox_norm_top_left:
        file_meta["bbox_norm_top_left"] = bbox_norm_top_left

    return {
        "doc_id": doc_id,
        "source_path": str(source_path.resolve()),
        "file_name": source_path.name,
        "page_number": page_number,
        "type": element_type,
        "text": normalized_text,
        "html": html,
        "image_path": image_path,
        "coordinates": coordinates,          # pixel-space bbox (kept for compatibility)
        "element_index": element_idx,
        "meta": file_meta                    # <-- now includes coord_system/layout_size/bbox_norm
    }


def parse_pdf_with_unstructured(pdf_path: Path, image_dir: Path, use_hires: bool = True) -> List[Element]:
    """Parse PDF using unstructured library with fallback strategy."""
    if not HAS_UNSTRUCTURED:
        raise ImportError("unstructured library not available")

    strategy = "hi_res" if use_hires else "fast"
    logger.debug(f"Parsing {pdf_path.name} with strategy='{strategy}'")

    image_dir.mkdir(parents=True, exist_ok=True)
    abs_image_dir = image_dir.resolve()
    abs_pdf_path = pdf_path.resolve()

    kwargs = {
        "filename": str(abs_pdf_path),
        "strategy": strategy,
        "languages": ["eng"],
        "extract_images_in_pdf": True,
        "image_output_dir_path": str(abs_image_dir),  # <-- absolute, not relative
    }
    if strategy == "hi_res":
        kwargs["infer_table_structure"] = True

    try:
        elements = partition_pdf(**kwargs)
        logger.info(f"Successfully parsed {pdf_path.name} using '{strategy}' strategy")
        return elements
    except Exception as e:
        if strategy == "hi_res":
            logger.warning(f"Hi-res parsing failed for {pdf_path.name}, falling back to 'fast': {e}")
            elements = partition_pdf(
                filename=str(abs_pdf_path),
                strategy="fast",
                languages=["eng"],
                extract_images_in_pdf=True,
                image_output_dir_path=str(abs_image_dir),  # <-- still absolute
            )
            logger.info(f"Successfully parsed {pdf_path.name} using 'fast' fallback strategy")
            return elements
        raise


def extract_tables_with_camelot(pdf_path: Path, output_dir: Path, doc_id: str) -> List[Dict[str, Any]]:
    """Extract tables using Camelot as fallback when no HTML tables found."""
    if not HAS_CAMELOT:
        logger.warning(f"camelot-py not available, skipping table fallback for {pdf_path.name}")
        return []
    
    table_dir = output_dir / f"{pdf_path.stem}_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    
    camelot_records = []
    
    try:
        # Try lattice flavor first
        tables = camelot.read_pdf(str(pdf_path), flavor="lattice", pages="all")
        
        # If no tables found, try stream flavor
        if len(tables) == 0:
            tables = camelot.read_pdf(str(pdf_path), flavor="stream", pages="all")
        
        for i, table in enumerate(tables):
            try:
                page_num = table.page
                markdown_filename = f"table_{page_num}_{i}.md"
                markdown_path = table_dir / markdown_filename
                
                # Convert to markdown
                df = table.df
                markdown_content = df.to_markdown(index=False)
                markdown_path.write_text(markdown_content, encoding='utf-8')
                
                # Create JSONL record
                record = {
                    "doc_id": doc_id,
                    "source_path": str(pdf_path.resolve()),
                    "file_name": pdf_path.name,
                    "page_number": page_num,
                    "type": "Table",
                    "text": df.to_string(index=False),
                    "html": "",
                    "image_path": "",
                    "coordinates": None,
                    "element_index": -1,  # Special marker for Camelot tables
                    "markdown_path": str(markdown_path.relative_to(output_dir.parent)),
                    "meta": get_file_metadata(pdf_path)
                }
                camelot_records.append(record)
                
            except Exception as e:
                logger.warning(f"Failed to process Camelot table {i} on page {table.page}: {e}")
                continue
        
        logger.info(f"Camelot extracted {len(camelot_records)} tables from {pdf_path.name}")
        
    except Exception as e:
        logger.warning(f"Camelot table extraction failed for {pdf_path.name}: {e}")
    
    return camelot_records


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


def write_jsonl_atomic(output_file: Path, records: List[Dict[str, Any]]) -> None:
    """Write JSONL file atomically using temporary file and rename."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.jsonl.tmp', 
        dir=output_file.parent, 
        delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        
        try:
            for record in records:
                json.dump(record, tmp_file, ensure_ascii=False)
                tmp_file.write('\n')
            
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            
            # Atomic rename
            tmp_path.rename(output_file)
            
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def find_pdf_files(src_dir: Path, recursive: bool = True) -> List[Path]:
    """Find all PDF files in source directory."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return list(src_dir.glob(pattern))


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
