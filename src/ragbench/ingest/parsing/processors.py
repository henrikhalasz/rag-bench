#!/usr/bin/env python3
"""
Element and coordinate processing for PDF parsing.
"""

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

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

from .utils import normalize_text, get_file_metadata

logger = logging.getLogger(__name__)


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
    coord_system, layout_w, layout_h, coordinates, bbox_norm_top_left = extract_coordinates(element, meta)
    
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


def extract_coordinates(element, meta):
    """Extract coordinate information from element or metadata."""
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
            
    return coord_system, layout_w, layout_h, coordinates, bbox_norm_top_left


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
        "image_output_dir_path": str(abs_image_dir),
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
                image_output_dir_path=str(abs_image_dir),
            )
            logger.info(f"Successfully parsed {pdf_path.name} using 'fast' fallback strategy")
            return elements
        raise
