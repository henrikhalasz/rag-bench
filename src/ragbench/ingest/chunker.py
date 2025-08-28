# chunker.py
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .schema import (
    SCHEMA_VERSION,
    ElementRecord,
    Block,
    Chunk,
    Manifest,
    NormalizedBBox,
    to_jsonl,
)

logger = logging.getLogger(__name__)

@dataclass
class SectionInfo:
    id: str
    level: int
    title: str
    path: str
    index: int  # ordinal within its level

# -------- Tokenizer abstraction ----------------------------------------------
# IMPLEMENTED: simple default tokenizer (word-based).
# EXTRA: plug in model tokenizer (e.g., tiktoken) via --tokenizer=.
TokenCounter = Callable[[str], int]

def simple_token_counter(text: str) -> int:
    """Approximate tokens; safe default across models."""
    # crude but stable: ~1 token per 0.75 words
    words = len(re.findall(r"\w+|\S", text))
    return max(1, int(round(words / 0.75)))

# --------- Tunables -----------------------------------------------------------
# MIN_CHUNK_CHARS: guardrail to prevent useless orphan chunks like "a." or "e.g."
# Keep char-based to avoid tokenizer coupling; raise to 20–30 if your corpus is noisy.
MIN_CHUNK_CHARS = 15
# EXTRA (not implemented): a token-based guard (e.g., MIN_CHUNK_TOKENS = 5) can be
# added once you wire a model tokenizer.

# --------- Load parser output -------------------------------------------------
def load_elements(jsonl_path: Path) -> List[ElementRecord]:
    """Load parser JSONL into ElementRecord objects."""
    out: List[ElementRecord] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(ElementRecord(
                element_index=d.get("element_index", -1),
                doc_id=d["doc_id"],
                file_name=d["file_name"],
                source_path=d["source_path"],
                page_number=d.get("page_number"),
                type=d.get("type", "Unknown"),
                text=d.get("text", "") or "",
                html=d.get("html", "") or "",
                image_path=d.get("image_path", "") or "",
                coordinates=d.get("coordinates"),
                meta=d.get("meta", {}),
            ))
    return out

# --------- Geometry normalization --------------------------------------------
def normalize_bbox(coords: Optional[Dict[str, float]],
                   page_wh: Tuple[float, float],
                   meta: Optional[Dict] = None) -> Optional[NormalizedBBox]:
    """
    Normalize to [0..1], top-left origin.
    Priority:
      1) Use parser-provided meta['bbox_norm_top_left'] if present.
      2) Else, use coords + authoritative layout_size (meta['layout_size']).
      3) Else, use coords + page_wh fallback.
    Assumes hi_res PixelSpace (top-left) by default; if a bottom-left system ever appears
    (e.g., PDF points), add a single y-flip here keyed off meta['coord_system'].
    """
    if meta:
        bn = meta.get("bbox_norm_top_left")
        if bn and all(k in bn for k in ("x1", "y1", "x2", "y2")):
            return NormalizedBBox(x1=float(bn["x1"]), y1=float(bn["y1"]),
                                  x2=float(bn["x2"]), y2=float(bn["y2"]))

    if not coords:
        return None

    # choose the scaling canvas
    w = h = None
    if meta:
        ls = meta.get("layout_size")
        if ls:
            w = float(ls.get("w") or 0.0)
            h = float(ls.get("h") or 0.0)
    if not (w and h and w > 0 and h > 0):
        w, h = page_wh

    if not (w and h and w > 0 and h > 0):
        return None

    x1 = coords.get("x1"); y1 = coords.get("y1")
    x2 = coords.get("x2"); y2 = coords.get("y2")
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    # PixelSpace top-left normalization (no Y flip!)
    nx1 = max(0.0, min(1.0, float(x1) / w))
    ny1 = max(0.0, min(1.0, float(y1) / h))
    nx2 = max(0.0, min(1.0, float(x2) / w))
    ny2 = max(0.0, min(1.0, float(y2) / h))
    if nx2 < nx1: nx1, nx2 = nx2, nx1
    if ny2 < ny1: ny1, ny2 = ny2, ny1
    return NormalizedBBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2)

def infer_page_wh(elements: List[ElementRecord]) -> Dict[int, Tuple[float, float]]:
    """
    Prefer authoritative layout size from parser meta; otherwise fall back to max coords.
    """
    out: Dict[int, Tuple[float, float]] = {}
    max_seen: Dict[int, Dict[str, float]] = defaultdict(lambda: {"x2": 0.0, "y2": 0.0})

    for el in elements:
        if not el.page_number:
            continue
        p = el.page_number

        # Prefer parser-provided layout size (PixelSpace for hi_res)
        layout = (el.meta or {}).get("layout_size")
        if layout and isinstance(layout, dict):
            w = float(layout.get("w") or 0.0)
            h = float(layout.get("h") or 0.0)
            if w > 0 and h > 0:
                out[p] = (w, h)
                continue  # page has an authoritative size; no need to track max

        # Fallback: derive from max observed coordinates
        if el.coordinates:
            max_seen[p]["x2"] = max(max_seen[p]["x2"], float(el.coordinates.get("x2", 0.0)))
            max_seen[p]["y2"] = max(max_seen[p]["y2"], float(el.coordinates.get("y2", 0.0)))

    # Fill remaining pages from max extents if no authoritative layout found
    for p, m in max_seen.items():
        if p not in out:
            w = max(1.0, m["x2"])
            h = max(1.0, m["y2"])
            out[p] = (w, h)

    return out


def filter_toc_elements(elements: List[ElementRecord]) -> List[ElementRecord]:
    """
    Robust Table of Contents filtering using multiple detection strategies.
    
    Strategy:
    1. Trust parser metadata first (e.g., "is_toc": true)
    2. Use positional heuristics (ToC typically in first 20% of document)
    3. Apply content-based pattern matching with improved detection
    4. Handle both standalone ToC entries and large ToC tables
    5. Filter document history tables that mention ToC restructuring
    """
    if not elements:
        return elements
    
    # Phase 1: Check for explicit parser signals
    explicit_toc_elements = set()
    for el in elements:
        if el.meta and el.meta.get("is_toc"):
            explicit_toc_elements.add(el.element_index)
    
    # Phase 2: Determine document extent for positional analysis
    total_pages = max((el.page_number or 0) for el in elements if el.page_number)
    if total_pages == 0:
        return elements
    
    # ToC typically occurs in first 20% of document, but at least first 5 pages
    toc_page_limit = max(5, int(total_pages * 0.20))
    
    # Phase 3: Analyze early pages for ToC patterns
    early_elements = [el for el in elements if (el.page_number or 0) <= toc_page_limit]
    
    # Enhanced content patterns for ToC detection
    toc_patterns = [
        # Explicit ToC titles
        re.compile(r'^\s*TABLE\s+OF\s+CONTENTS\s*$', re.IGNORECASE),
        re.compile(r'^\s*LIST\s+OF\s+(FIGURES|TABLES|APPENDICES)\s*$', re.IGNORECASE),
        
        # Dot leaders: text ending with dots and page number
        re.compile(r'.*\.{3,}\s*\d+\s*$'),
        
        # Page number after significant whitespace or dots
        re.compile(r'.*\s{5,}\d{1,4}\s*$'),  # 5+ spaces before page number
        re.compile(r'.*\.{2,}\s*\d{1,4}\s*$'), # 2+ dots before page number
        
        # Tab-separated content ending in number
        re.compile(r'.*\t+.*\d{1,4}\s*$'),
        
        # Common ToC prefixes with numbers
        re.compile(r'^\s*(Chapter|Section|Part|Appendix|Figure|Table)\s+[\d\w]', re.IGNORECASE),
        
        # Numbered section patterns
        re.compile(r'^\s*\d+(\.\d+)*\s+[A-Z]', re.IGNORECASE),  # "1.2 HEADING"
        re.compile(r'^\s*[A-Z]+(\.\d+)*\s+[A-Z]', re.IGNORECASE),  # "A.1 HEADING"
    ]
    
    # Phase 4: Enhanced content analysis with multiple detection methods
    toc_candidates = set()
    page_toc_density = defaultdict(int)
    large_toc_elements = set()  # Track potentially large ToC tables
    
    for el in early_elements:
        if not el.text or not el.page_number:
            continue
            
        text_clean = re.sub(r'\s+', ' ', el.text.strip())
        original_text = el.text.strip()
        
        # Method 1: Direct ToC title detection (always filter these)
        if re.match(r'^\s*TABLE\s+OF\s+CONTENTS\s*$', text_clean, re.IGNORECASE):
            toc_candidates.add(el.element_index)
            page_toc_density[el.page_number] += 1
            logger.debug(f"Found ToC title: '{text_clean}'")
            continue
        
        # Method 2: Large ToC tables (often contain full ToC in one element)
        if (el.type and el.type.lower() == "table" and 
            len(original_text) > 500 and  # Large table content
            ("TABLE OF CONTENTS" in original_text.upper() or
             "LIST OF FIGURES" in original_text.upper() or
             "LIST OF TABLES" in original_text.upper() or
             original_text.count('....') > 5 or  # Many dot leaders
             len(re.findall(r'\d+\s*$', original_text, re.MULTILINE)) > 10)):  # Many page numbers
            large_toc_elements.add(el.element_index)
            logger.debug(f"Found large ToC table: {len(original_text)} chars")
            continue
        
        # Method 3: Document history tables mentioning ToC changes
        if (el.type and el.type.lower() == "table" and 
            len(original_text) > 200 and
            ("Table of Contents" in original_text or
             "List of Figures" in original_text or
             "List of Tables" in original_text or
             "Chapter " in original_text) and
            ("renumber" in original_text.lower() or
             "revise" in original_text.lower() or
             "insert" in original_text.lower())):
            toc_candidates.add(el.element_index)
            page_toc_density[el.page_number] += 1
            logger.debug(f"Found document history table mentioning ToC")
            continue
        
        # Method 4: Standard ToC entry patterns (keep existing logic for shorter entries)
        if len(text_clean) <= 200:  # Standard ToC entries
            is_toc_like = False
            
            # Pattern matching
            for pattern in toc_patterns:
                if pattern.match(text_clean):
                    is_toc_like = True
                    break
            
            # Additional heuristics for numbered sections
            if not is_toc_like and len(text_clean.split()) <= 15:
                # Check for section numbering patterns
                if re.match(r'^\s*(\d+\.?\d*|\w+\.?\d*)\s+\S', text_clean):
                    # Additional check: does it end with a number (page reference)?
                    if re.search(r'\b\d{1,4}\s*$', text_clean):
                        is_toc_like = True
                
                # Check for high density of dots (dot leaders)
                if text_clean.count('.') > len(text_clean.split()) and text_clean.count('.') > 5:
                    is_toc_like = True
            
            if is_toc_like:
                toc_candidates.add(el.element_index)
                page_toc_density[el.page_number] += 1
    
    # Phase 5: Enhanced validation using multiple criteria
    validated_toc_elements = set(explicit_toc_elements)  # Always trust parser
    validated_toc_elements.update(large_toc_elements)    # Always filter large ToC tables
    
    # Validate regular ToC candidates by page density
    for page, density in page_toc_density.items():
        total_elements_on_page = sum(1 for el in early_elements if el.page_number == page and el.text)
        
        if total_elements_on_page > 0:
            toc_ratio = density / total_elements_on_page
            
            # More aggressive filtering: lower threshold for ToC-heavy pages
            if (toc_ratio > 0.3 and density >= 2) or density >= 5:  # Either 30%+ ratio or 5+ ToC elements
                for el in early_elements:
                    if (el.page_number == page and 
                        el.element_index in toc_candidates):
                        validated_toc_elements.add(el.element_index)
    
    # Phase 6: Apply filtering with enhanced logging
    filtered_elements = []
    stats = {"toc_filtered": 0, "explicit_toc": len(explicit_toc_elements), "large_tables": len(large_toc_elements)}
    
    for el in elements:
        if el.element_index in validated_toc_elements:
            stats["toc_filtered"] += 1
            element_type = "explicit" if el.element_index in explicit_toc_elements else \
                          "large_table" if el.element_index in large_toc_elements else \
                          "pattern"
            logger.debug(f"Filtered ToC element ({element_type}): '{(el.text or '')[:50]}...'")
            continue
        filtered_elements.append(el)
    
    # Log filtering statistics
    original_count = len(elements)
    filtered_count = len(filtered_elements)
    logger.info(f"ToC filtering: {original_count} -> {filtered_count} elements "
                f"(removed {original_count - filtered_count}: "
                f"{stats['explicit_toc']} explicit, "
                f"{stats['large_tables']} large tables, "
                f"{stats['toc_filtered'] - stats['explicit_toc'] - stats['large_tables']} pattern-based)")
    
    return filtered_elements


def filter_headers_and_footers(
    elements: List[ElementRecord], 
    page_wh: Dict[int, Tuple[float, float]]
) -> List[ElementRecord]:
    """
    Production-grade header/footer filtering using position, repetition, and content analysis.
    
    Strategy:
    1. Identify repeating text patterns across multiple pages
    2. Filter by vertical position (top/bottom margins)
    3. Apply content-based heuristics (page numbers, short repetitive text)
    4. Handle edge cases for legitimate content in margin areas
    """
    if not elements:
        return elements
    
    # Phase 1: Analyze text repetition patterns across pages
    text_page_map = defaultdict(set)  # text -> set of pages where it appears
    text_positions = defaultdict(list)  # text -> list of (page, normalized_y_center) tuples
    
    for el in elements:
        if not el.page_number or not el.text:
            continue
            
        text_clean = re.sub(r'\s+', ' ', el.text.strip())
        if not text_clean or len(text_clean) > 150:  # Skip very long text (unlikely headers/footers)
            continue
            
        text_page_map[text_clean].add(el.page_number)
        
        # Calculate normalized vertical position
        bbox = normalize_bbox(el.coordinates, page_wh.get(el.page_number, (1.0, 1.0)), el.meta)
        if bbox:
            y_center = (bbox.y1 + bbox.y2) / 2
            text_positions[text_clean].append((el.page_number, y_center))
    
    # Phase 2: Identify boilerplate text (appears on multiple pages)
    total_pages = len(set(el.page_number for el in elements if el.page_number))
    min_pages_for_boilerplate = max(2, total_pages // 3)  # Adaptive threshold
    
    boilerplate_texts = set()
    for text, pages in text_page_map.items():
        if len(pages) >= min_pages_for_boilerplate:
            # Additional check: consistent vertical positioning
            positions = text_positions[text]
            y_positions = [y for _, y in positions]
            if len(y_positions) > 1:
                y_std = (sum((y - sum(y_positions)/len(y_positions))**2 for y in y_positions) / len(y_positions))**0.5
                # If text appears in consistent vertical positions, it's likely boilerplate
                if y_std < 0.05:  # Very consistent positioning
                    boilerplate_texts.add(text)
            else:
                boilerplate_texts.add(text)
    
    # Phase 3: Define margin thresholds
    HEADER_THRESHOLD = 0.12  # Top 12% of page
    FOOTER_THRESHOLD = 0.88  # Bottom 12% of page
    
    # Phase 4: Apply filtering logic
    filtered_elements = []
    stats = {"headers_filtered": 0, "footers_filtered": 0, "boilerplate_filtered": 0}
    
    for el in elements:
        if not el.page_number:
            filtered_elements.append(el)
            continue

        # --- NEW: never drop real figures/images that have a file on disk ---
        if (el.type or "").lower() in ("image", "figure") and (el.image_path or "").strip():
            filtered_elements.append(el)
            continue
        # --------------------------------------------------------------------
            
        text_clean = re.sub(r'\s+', ' ', (el.text or '').strip())
        
        # Skip empty or whitespace-only *text* elements; images are already preserved above
        if not text_clean:
            continue
            
        # Rule 1: Filter identified boilerplate text
        if text_clean in boilerplate_texts:
            stats["boilerplate_filtered"] += 1
            logger.debug(f"Filtered boilerplate: '{text_clean[:50]}...'")
            continue
        
        # Rule 2: Position-based filtering with content validation
        bbox = normalize_bbox(el.coordinates, page_wh.get(el.page_number, (1.0, 1.0)), el.meta)
        if bbox:
            y_center = (bbox.y1 + bbox.y2) / 2
            
            # Check if in header/footer zones
            in_header_zone = y_center < HEADER_THRESHOLD
            in_footer_zone = y_center > FOOTER_THRESHOLD
            
            if in_header_zone or in_footer_zone:
                # Additional content-based checks for margin elements
                should_filter = False
                
                # Pattern: Page numbers (e.g., "1 of 10", "Page 5", "5")
                if re.match(r'^\s*(\d+\s+(of|/)\s+\d+|page\s+\d+|\d+)\s*$', text_clean, re.IGNORECASE):
                    should_filter = True
                
                # Pattern: Document identifiers (e.g., "NASA-STD-8739.4A", "REPORT-123")
                elif re.match(r'^[A-Z]{2,}-[A-Z0-9\-\.]+[A-Z0-9]$', text_clean):
                    should_filter = True
                
                # Pattern: Short repetitive text (likely headers/footers)
                elif len(text_clean) < 50 and len(text_clean.split()) <= 5:
                    # Check if this short text pattern appears on multiple pages
                    if len(text_page_map[text_clean]) > 1:
                        should_filter = True
                
                # Pattern: Copyright notices, URLs, etc.
                elif any(pattern in text_clean.lower() for pattern in 
                        ['copyright', '©', 'confidential', 'proprietary', 'www.', 'http']):
                    should_filter = True
                
                if should_filter:
                    if in_header_zone:
                        stats["headers_filtered"] += 1
                    else:
                        stats["footers_filtered"] += 1
                    logger.debug(f"Filtered {'header' if in_header_zone else 'footer'}: '{text_clean[:50]}...'")
                    continue
        
        # Rule 3: Filter standalone page references that might be in content area
        if re.match(r'^\s*\d+\s+of\s+\d+\s*$', text_clean, re.IGNORECASE):
            stats["footers_filtered"] += 1
            continue
        
        # Keep element if it passed all filters
        filtered_elements.append(el)
    
    # Log filtering statistics
    original_count = len(elements)
    filtered_count = len(filtered_elements)
    logger.info(f"Header/footer filtering: {original_count} -> {filtered_count} elements "
                f"(removed {original_count - filtered_count}: "
                f"{stats['headers_filtered']} headers, "
                f"{stats['footers_filtered']} footers, "
                f"{stats['boilerplate_filtered']} boilerplate)")
    
    return filtered_elements

# --------- Block building / roles --------------------------------------------
FIG_CAP_RE = re.compile(r"^\s*Figure\s+([\dA-Za-z\-\.]+)", re.IGNORECASE)
TAB_CAP_RE = re.compile(r"^\s*Table\s+([\dA-Za-z\-\.]+)", re.IGNORECASE)
HEAD_RE    = re.compile(r"^(\d+(\.\d+){0,3})\s+\S+")

def classify_role(el_type: str, text: str) -> str:
    """
    Heuristic role assignment, but **trust the parser first**.
    Unstructured emits types like: Title, NarrativeText, ListItem, Table, Image, Header, Footer.
    We map those directly; only if the parser's type isn't decisive do we fall back to regexes.
    """
    t = (el_type or "").lower()
    # 1) Parser-led mapping
    if t in ("title", "h1", "h2", "h3"):      return t
    if t in ("table",):                       return "table"
    if t in ("image", "figure"):              return "figure"
    if t in ("figurecaption", "tablecaption","caption"): return "caption"
    if t in ("narrativetext", "list", "listitem", "text"): return "paragraph"
    if t in ("header", "footer"):             return "paragraph"  # we filter repetitive boilerplate later

    # 2) Fallbacks when parser type isn't decisive
    if FIG_CAP_RE.match(text) or TAB_CAP_RE.match(text):
        return "caption"
    if HEAD_RE.match(text) and len(text) < 200:
        return "h2"
    # very short all-caps-ish lines can be headings
    letters = re.sub(r"\W", "", text or "")
    if 3 <= len(text) <= 80 and letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return "h2"
    return "paragraph"

def heading_level(role: str, text: str) -> Optional[int]:
    """
    Determine heading level using a production-grade Hierarchy of Signals.

    This function is designed to be precise by trusting explicit signals first,
    and robust by using intelligent heuristics as fallbacks.
    """
    s = (text or "").strip()
    t = (role or "").lower()

    # --- Rule 0 (Guard Clauses): Immediately reject non-headings ---
    # The parser is very reliable at identifying list items. If it's a list,
    # it's not a heading, regardless of numbering. This is our most important filter.
    if t == "listitem":
        return None

    # --- Hierarchy of Signals ---

    # Rule 1 (Most Precise): Numbered Headings
    # This is the strongest signal in technical documents and reports.
    # We match patterns like "8.", "8.1", "8.1.1".
    m = re.match(r'^\s*(\d+(?:\.\d+)*)\b', s)
    if m:
        # CRITICAL HEURISTIC: Check word count to distinguish a heading
        # from a numbered paragraph. A heading is almost always short.
        word_count = len(re.findall(r"\w+", s))
        if word_count <= 25:
            # The number of dot-separated segments gives the level.
            level = s.count('.') + 1
            return level
        else:
            # It's a numbered paragraph, not a heading.
            return None

    # Rule 2 (High Confidence): Parser-Assigned Title Roles
    # If the parser's vision model identifies a clear title, we trust it.
    if t in ("title", "h1"):
        return 1

    # Rule 3 (Good Heuristic): Typographical Cues for Un-numbered Headings
    # This catches chapter titles or major sections in less formal documents.
    words = re.findall(r"\b\w+\b", s)
    if 1 <= len(words) <= 15:
        letters = re.findall(r"[A-Za-z]", s)
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            # A high ratio of capital letters in a short line is a strong signal.
            if upper_ratio >= 0.70:
                return 2 # Default to level 2 for stylistic headings

    # If no rules match, it's not a heading.
    return None

def merge_adjacent_text(elements: List[ElementRecord],
                        page_wh: Dict[int, Tuple[float, float]]) -> List[Block]:
    """
    IMPLEMENTED: merge contiguous text runs per page & column into paragraphs,
    keep tables and figures atomic. Captions remain their own blocks (to glue later).
    """
    # Simple column detection via x-center clustering (1 or 2 columns).
    per_page: Dict[int, List[ElementRecord]] = defaultdict(list)
    for el in elements:
        if el.page_number:
            per_page[el.page_number].append(el)

    blocks: List[Block] = []
    bid = 0
    for page, els in per_page.items():
        w, h = page_wh.get(page, (1.0,1.0))
        # Estimate a vertical tolerance from normalized heights
        def est_line_h_norm(arr: List[ElementRecord]) -> float:
            hs = []
            for e in arr:
                if e.coordinates:
                    b = normalize_bbox(e.coordinates, page_wh.get(page, (1.0,1.0)), e.meta)
                    if b:
                        hs.append(b.y2 - b.y1)
            hs.sort()
            return max(0.006, hs[len(hs)//2] * 0.6 if hs else 0.012)  # ~ (4–16 px on a 1500px page
        tol = est_line_h_norm(els)

        def key(e: ElementRecord):
            b = normalize_bbox(e.coordinates, page_wh.get(page, (1.0,1.0)), e.meta)
            y = b.y1 if b else 0.0
            x = b.x1 if b else 0.0
            ybin = round(y / tol) if tol > 0 else 0
            return (ybin, x)
        els.sort(key=key)

        carry_text = []
        carry_ids: List[int] = []
        carry_bbox: Optional[NormalizedBBox] = None

        def flush_paragraph():
            nonlocal bid, carry_text, carry_ids, carry_bbox
            text = " ".join(s.strip() for s in carry_text if s.strip())
            if not text: 
                carry_text, carry_ids, carry_bbox = [], [], None
                return
            blocks.append(Block(
                block_id=f"b{bid}", doc_id=els[0].doc_id, page=page,
                role="paragraph", text=text, source_element_ids=carry_ids[:],
                bbox_norm=carry_bbox
            ))
            bid += 1
            carry_text, carry_ids, carry_bbox = [], [], None

        for el in els:
            role = classify_role(el.type, el.text or "")
            bbox = normalize_bbox(el.coordinates, page_wh.get(page, (1.0,1.0)), el.meta)
            if role in ("table", "figure", "caption", "title", "h2"):
                flush_paragraph()
                blocks.append(Block(
                    block_id=f"b{bid}", doc_id=el.doc_id, page=page, role=role,
                    text=(el.text or ""), html=(el.html or ""), image_path=(el.image_path or ""),
                    source_element_ids=[el.element_index], bbox_norm=bbox
                ))
                bid += 1
            else:
                # paragraph-ish: merge
                carry_text.append(el.text or "")
                carry_ids.append(el.element_index)
                carry_bbox = bbox if carry_bbox is None else carry_bbox.union(bbox)  # type: ignore

        flush_paragraph()

    return blocks

# --------- Caption sticking ---------------------------------------------------
def attach_captions(blocks: List[Block]) -> List[Block]:
    """
    IMPLEMENTED: attach nearest caption on same page to preceding figure/table.
    EXTRA: distance thresholds, multi-page tolerance.
    """
    by_page: Dict[int, List[Block]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)

    out: List[Block] = []
    for page, arr in by_page.items():
        i = 0
        while i < len(arr):
            b = arr[i]
            if b.role == "caption":
                # Look back for nearest figure/table
                j = i - 1
                attached = False
                while j >= 0:
                    if arr[j].role in ("figure", "table"):
                        # attach caption text
                        if arr[j].text:
                            arr[j].text = f"{arr[j].text}\n{b.text}"
                        else:
                            arr[j].text = b.text
                        attached = True
                        break
                    j -= 1
                if not attached:
                    out.append(b)  # orphan caption
            else:
                out.append(b)
            i += 1
    return out

def coalesce_semantic_blocks(blocks: List[Block]) -> List[Block]:
    """
    Merge micro-blocks into richer semantic units:
      - Heading + following paragraph (same page)
      - Caption -> preceding figure/table (same page)

    Idempotent with attach_captions; unions bbox and source IDs.
    O(n) single pass; preserves original ordering otherwise.
    """
    out: List[Block] = []
    i = 0

    def _union_bbox(a: Optional[NormalizedBBox], b: Optional[NormalizedBBox]) -> Optional[NormalizedBBox]:
        if a and b:
            return a.union(b)  # type: ignore
        return a or b

    while i < len(blocks):
        b = blocks[i]

        # Rule B: merge caption into preceding figure/table (defensive; attach_captions already handles text)
        if b.role == "caption" and out and out[-1].page == b.page and out[-1].role in ("figure", "table"):
            prev = out[-1]
            # merge text/html
            if b.text:
                prev.text = (prev.text + "\n" + b.text).strip() if prev.text else b.text
            if b.html and not prev.html:
                prev.html = b.html
            # merge ids/bbox
            prev.source_element_ids = (prev.source_element_ids or []) + (b.source_element_ids or [])
            prev.bbox_norm = _union_bbox(prev.bbox_norm, b.bbox_norm)
            # drop the caption block
            i += 1
            continue

        # Rule A: heading + following paragraph (same page)
        if heading_level(b.role, b.text) is not None:
            if i + 1 < len(blocks):
                nxt = blocks[i + 1]
                if nxt.page == b.page and nxt.role == "paragraph":
                    # merge into the heading block; keep heading role for sectioning
                    b.text = ((b.text or "").rstrip() + "\n" + (nxt.text or "").lstrip()).strip()
                    if nxt.html and not b.html:
                        b.html = nxt.html
                    b.source_element_ids = (b.source_element_ids or []) + (nxt.source_element_ids or [])
                    b.bbox_norm = _union_bbox(b.bbox_norm, nxt.bbox_norm)
                    out.append(b)
                    i += 2
                    continue

        out.append(b)
        i += 1

    return out

def assign_sections(blocks: List[Block]) -> Dict[str, SectionInfo]:
    """
    Build section hierarchy from heading blocks and assign to all blocks.

    Stateful, stack-based approach:
      - Maintain a stack of current parent headings.
      - On a heading: pop to its parent level, then push a new SectionInfo.
      - On non-heading: inherit the section from the top of the stack.
    """
    section_stack: List[SectionInfo] = []
    # Track ordinal index within each (level, parent) group
    per_parent_counts: Dict[Tuple[int, Optional[str]], int] = defaultdict(int)

    sections: List[SectionInfo] = []
    map_block_to_sec: Dict[str, SectionInfo] = {}

    for b in blocks:
        lvl = heading_level(b.role, b.text)
        if lvl is not None:
            # Trim stack so that the new heading becomes a child of the top element
            while section_stack and section_stack[-1].level >= lvl:
                section_stack.pop()

            title = (b.text or "").strip().splitlines()[0][:120]
            parent_titles = [s.title for s in section_stack]
            path = " > ".join(parent_titles + [title]) if parent_titles else title

            parent_id = section_stack[-1].id if section_stack else None
            per_parent_counts[(lvl, parent_id)] += 1
            idx = per_parent_counts[(lvl, parent_id)]

            sid = f"s{len(sections)}"
            sec = SectionInfo(id=sid, level=lvl, title=title, path=path, index=idx)
            sections.append(sec)
            section_stack.append(sec)
            map_block_to_sec[b.block_id] = sec
        else:
            # Non-heading blocks inherit current section (if any)
            if section_stack:
                map_block_to_sec[b.block_id] = section_stack[-1]

    return map_block_to_sec

# --------- Helper functions for chunk improvements -----------------------------
def get_image_path_for_figure(element_ids: List[int], elements_by_id: Dict[int, ElementRecord]) -> str:
    """Look up the first Image element in source_element_ids and return its image_path."""
    for el_id in element_ids:
        el = elements_by_id.get(el_id)
        if el and el.type.lower() == "image" and el.image_path:
            return el.image_path
    return ""

def compute_absolute_bbox(normalized_bbox: Optional[NormalizedBBox], 
                         page_width: float, page_height: float) -> Optional[Dict[str, float]]:
    """Convert normalized bbox to absolute coordinates."""
    if not normalized_bbox:
        return None
    return {
        "x1": normalized_bbox.x1 * page_width,
        "y1": normalized_bbox.y1 * page_height,
        "x2": normalized_bbox.x2 * page_width,
        "y2": normalized_bbox.y2 * page_height,
    }

# --------- Chunk assembly -----------------------------------------------------
def assemble_chunks(blocks: List[Block],
                    elements: List[ElementRecord],
                    page_wh: Dict[int, Tuple[float, float]],
                    max_tokens: int = 900,
                    overlap_tokens: int = 120,
                    tok: TokenCounter = simple_token_counter,
                    section_map: Optional[Dict[str, SectionInfo]] = None) -> List[Chunk]:
    """
    IMPLEMENTED:
      - Keep tables and figures atomic (no overlap).
      - Keep title/heading glued to its immediate paragraph if budget allows.
      - For long prose, create overlapping windows.
      - Set image_path for figure chunks.
      - Add coordinate system fields to all chunks.
      - Section-aware windowing: don't cross section boundaries.
    EXTRA:
      - Parent/child strategy, sentence-window adjacency, profile presets.
    """
    # Create element lookup for image path resolution
    elements_by_id = {el.element_index: el for el in elements}
    
    # Track how many chunks we've emitted per section
    sec_chunk_counts = defaultdict(int)
    
    chunks: List[Chunk] = []
    cid_prev: Optional[str] = None

    i = 0
    while i < len(blocks):
        b = blocks[i]
        cur_sec = section_map.get(b.block_id) if section_map else None

        # --- Guardrail: eliminate tiny orphan chunks (Issue #1) ----------------
        # Skip paragraph-ish blocks that are too small to be useful on their own.
        # (We still allow tiny headings if they glue to a following paragraph below.)
        if b.role in ("paragraph", "other"):
            if len((b.text or "").strip()) < MIN_CHUNK_CHARS:
                i += 1
                continue
        # ----------------------------------------------------------------------

        def new_chunk(text: str, html: str, ids: List[int], bbox: Optional[NormalizedBBox],
                      role: str, page: int) -> Chunk:
            cid = f"c{len(chunks)}"
            
            # Get page dimensions (fall back to inferred if no MediaBox)
            page_width, page_height = page_wh.get(page, (1.0, 1.0))
            
            # Compute absolute bbox
            bbox_abs = compute_absolute_bbox(bbox, page_width, page_height)
            
            # Get image path for figures
            img_path = ""
            if role == "figure":
                img_path = get_image_path_for_figure(ids, elements_by_id)
            
            ch = Chunk(
                chunk_id=cid,
                doc_id=b.doc_id,
                source_file="",  # filled later
                page_start=page, page_end=page,
                role=role, text=text, html=html,
                image_path=img_path,
                bbox_union_norm=bbox, 
                coord_system="top-left",
                page_width_pts=page_width,
                page_height_pts=page_height,
                bbox_union_abs=bbox_abs,
                source_element_ids=ids,
                prev_id=chunks[-1].chunk_id if chunks else None,
            )
            
            # Annotate section metadata
            if cur_sec:
                sec_chunk_counts[cur_sec.id] += 1
                ch.section_id = cur_sec.id
                ch.section_path = cur_sec.path
                ch.section_level = cur_sec.level
                ch.section_index = sec_chunk_counts[cur_sec.id]
            
            if chunks:
                chunks[-1].next_id = cid
            ch.finalize()
            return ch

        if b.role in ("table", "figure"):
            # atomic
            ch = new_chunk(b.text, b.html, b.source_element_ids, b.bbox_norm, b.role, b.page)
            ch.source_file = ""  # optional: fill with blocks’ meta
            chunks.append(ch)
            i += 1
            continue

        # prose block(s): try to glue a heading before/after
        start_i = i
        glue_prefix = []
        glue_ids: List[int] = []
        glue_bbox: Optional[NormalizedBBox] = None

        # include preceding title/h2 if immediately adjacent
        if (b.role in ("title", "h2") and (i + 1) < len(blocks) 
            and blocks[i+1].page == b.page 
            and blocks[i+1].role not in ("table", "figure")):
            glue_prefix.append(b.text)
            glue_ids += b.source_element_ids
            glue_bbox = b.bbox_norm
            i += 1
            b = blocks[i]

        text = "\n".join(glue_prefix + [b.text])
        ids = glue_ids + b.source_element_ids
        bbox = glue_bbox.union(b.bbox_norm) if glue_bbox and b.bbox_norm else (b.bbox_norm or glue_bbox)

        # windowing for long prose across subsequent paragraph blocks on same page
        # but DON'T cross sections
        j = i + 1
        while (j < len(blocks) 
               and blocks[j].role == "paragraph" 
               and blocks[j].page == b.page
               and (section_map.get(blocks[j].block_id) == cur_sec if section_map else True)):
            prospective = text + "\n" + blocks[j].text
            if tok(prospective) > max_tokens:
                break
            text = prospective
            ids += blocks[j].source_element_ids
            bbox = bbox.union(blocks[j].bbox_norm) if (bbox and blocks[j].bbox_norm) else (bbox or blocks[j].bbox_norm)
            j += 1

        # emit chunk
        ch = new_chunk(text, "", ids, bbox, role="paragraph", page=b.page)
        chunks.append(ch)

        # overlap for next window if we stopped early due to budget
        if (j < len(blocks) 
            and blocks[j].role == "paragraph" 
            and blocks[j].page == b.page
            and (section_map.get(blocks[j].block_id) == cur_sec if section_map else True)):
            # compute overlap by tokens (approx)
            toks = text.split()
            keep = max(0, min(len(toks), overlap_tokens))
            # prepare next block start by synthesizing a carry paragraph block
            carry_text = " ".join(toks[-keep:]) if keep else ""
            if carry_text:
                synthetic = Block(
                    block_id="__carry__", doc_id=b.doc_id, page=b.page, role="paragraph",
                    text=carry_text, source_element_ids=[], bbox_norm=bbox
                )
                blocks.insert(j, synthetic)
        i = max(j, i + 1)

    return chunks

# --------- Orchestration ------------------------------------------------------
def build_manifest(doc_id: str, source_file: str,
                   page_wh: Dict[int, Tuple[float, float]],
                   blocks: List[Block],
                   chunks: List[Chunk]) -> Manifest:
    counts = defaultdict(int)
    for b in blocks:
        counts[f"blocks_{b.role}"] += 1
    for c in chunks:
        counts[f"chunks_{c.role}"] += 1
    pages = {p: {"width": w, "height": h} for p, (w, h) in page_wh.items()}
    return Manifest(doc_id=doc_id, source_file=source_file, pages=pages, counts=dict(counts))

def normalize_section_metadata(chunks: List[Chunk], max_level: int = 6) -> None:
    """
    Sanitizes section metadata in-place, just before serialization.

    - Strips stray parenthetical codes like "(JWL4)" from section_path segments.
    - Normalizes the hierarchy delimiter to ' > ' and removes empty segments.
    - Clamps section_level into [1, max_level].
    """
    # e.g., "(JWL4)" "(ABC12)" "(NASA-01)" → removed
    STRAY_CODE_PAREN = re.compile(r"\(([A-Z]{2,}[A-Z0-9\-]*\d+[A-Z0-9\-]*)\)")
    SEP = re.compile(r"\s*>\s*")

    for c in chunks:
        # --- Clean path ---
        sp = getattr(c, "section_path", None)
        if sp:
            # split on any kind of ">" spacing the pipeline might have produced
            parts = [p.strip() for p in SEP.split(sp) if p.strip()]
            cleaned = []
            for seg in parts:
                # remove stray codes like (JWL4); keep human parentheses like "(continued)"
                seg2 = STRAY_CODE_PAREN.sub("", seg)
                # collapse whitespace, trim punctuation noise
                seg2 = re.sub(r"\s+", " ", seg2).strip(" -–—:; \t")
                if seg2:
                    cleaned.append(seg2)
            c.section_path = " > ".join(cleaned) if cleaned else None

        # --- Clamp level ---
        lvl = getattr(c, "section_level", None)
        if isinstance(lvl, int):
            c.section_level = max(1, min(max_level, lvl))

def run_chunking(input_jsonl: Path, out_dir: Path,
                 max_tokens: int = 900, overlap_tokens: int = 120,
                 tokenizer: TokenCounter = simple_token_counter) -> Tuple[List[Chunk], Manifest]:
    elements = load_elements(input_jsonl)
    if not elements:
        raise RuntimeError(f"No elements in {input_jsonl}")

    doc_id = elements[0].doc_id
    source_file = elements[0].file_name

    # Apply ToC filtering first, before any other processing
    logger.info(f"Filtering Table of Contents from {len(elements)} elements...")
    elements = filter_toc_elements(elements)

    page_wh = infer_page_wh(elements)
    
    # Apply header/footer filtering before block construction
    logger.info(f"Filtering headers and footers from {len(elements)} elements...")
    elements = filter_headers_and_footers(elements, page_wh)
    
    blocks = merge_adjacent_text(elements, page_wh)
    blocks = attach_captions(blocks)
    blocks = coalesce_semantic_blocks(blocks)   # <-- NEW
    section_map = assign_sections(blocks)
    chunks = assemble_chunks(blocks, elements, page_wh, max_tokens=max_tokens, overlap_tokens=overlap_tokens, tok=tokenizer, section_map=section_map)
    
    # Assign stable reading order per (page → y-bin → x), then by emission order as tie-break.
    # Compute per-page vertical tolerance from chunk heights
    def _page_tol(chunks, page):
        hs = []
        for c in chunks:
            if c.page_start == page and c.bbox_union_norm:
                b = c.bbox_union_norm
                hs.append(max(1e-6, b.y2 - b.y1))
        med = median(hs) if hs else 0.012
        return max(0.006, med * 0.6)

    tol_by_page = {p: _page_tol(chunks, p) for p in sorted({c.page_start for c in chunks})}

    def _order_key(c):
        b = c.bbox_union_norm
        tol = tol_by_page.get(c.page_start, 0.012)
        if not b:
            return (c.page_start, 9_999, 9_999)
        ybin = round(b.y1 / tol)
        return (c.page_start, ybin, b.x1)

    ordered = sorted(chunks, key=_order_key)
    for idx, ch in enumerate(ordered):
        ch.order_rank = idx
    # preserve prev/next you already set; order_rank is the explicit "truth"
    
    # fill source_file post-hoc
    for c in chunks:
        c.source_file = source_file

    # NEW: final cleanup pass on section metadata
    normalize_section_metadata(chunks, max_level=6)

    manifest = build_manifest(doc_id, source_file, page_wh, blocks, chunks)
    
    # Record coordinate contract and embedding policy
    manifest.notes.update({
        "embed_policy": "roles=title,h1,h2,h3,paragraph,list_item,caption,table|min_len=15|skip_empty_figures",
        "coord_contract": "normalized_top_left",
    })
    
    return chunks, manifest

# --------- CLI ---------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Layout-aware chunker")
    ap.add_argument("--input", type=Path, required=True, help="Parser JSONL file")
    ap.add_argument("--out", type=Path, default=Path("data/processed"), help="Output directory")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--overlap-tokens", type=int, default=120)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    chunks, manifest = run_chunking(args.input, args.out, args.max_tokens, args.overlap_tokens)

    # outputs
    to_jsonl(str(args.out / f"{manifest.doc_id}_chunks.jsonl"), chunks)
    with open(args.out / f"{manifest.doc_id}_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "doc_id": manifest.doc_id,
            "source_file": manifest.source_file,
            "coord_system": manifest.coord_system,
            "pages": manifest.pages,
            "counts": manifest.counts,
            "notes": manifest.notes,
        }, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d chunks for %s", len(chunks), manifest.doc_id)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
