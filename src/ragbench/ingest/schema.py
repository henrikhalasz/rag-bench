"""
Define data contracts for chunking.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Literal, Tuple
from datetime import datetime
import hashlib
import json

SCHEMA_VERSION = "chunk-v1"

# ---- Element types from parser ------------------------------------------------
# All major types from Unstructured, plus a fallback
ElementType = Literal[
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",
    "Image",
    "FigureCaption",
    "Header",
    "Footer",
    "Address",
    "Formula",
    "PageBreak",
    "UncategorizedText",
    "Unknown" # fallback
]
RoleType    = Literal["title", "h1", "h2", "h3", "paragraph", "list_item", "caption", "table", "figure", "other"]

@dataclass
class NormalizedBBox:
    """Unitless page-relative box in [0..1] coordinates with y=top origin."""
    x1: float
    y1: float
    x2: float
    y2: float

    def union(self, other: "NormalizedBBox") -> "NormalizedBBox":
        return NormalizedBBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

@dataclass
class ElementRecord:
    """Directly loaded from parser JSONL (one per element)."""
    element_index: int
    doc_id: str
    file_name: str
    source_path: str
    page_number: Optional[int]
    type: ElementType
    text: str
    html: str
    image_path: str
    coordinates: Optional[Dict[str, float]]  # parser-native units
    meta: Dict[str, object] = field(default_factory=dict)

# ---- Internal block built by the chunker -------------------------------------
@dataclass
class Block:
    """Merged, semantic unit (paragraph, heading, table, figure w/opt caption)."""
    block_id: str
    doc_id: str
    page: int
    role: RoleType
    text: str = ""
    html: str = ""
    image_path: str = ""
    source_element_ids: List[int] = field(default_factory=list)
    bbox_norm: Optional[NormalizedBBox] = None
    table_id: Optional[str] = None
    figure_id: Optional[str] = None

# ---- Final retrieval chunk ----------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page_start: int
    page_end: int
    role: RoleType
    text: str
    html: str = ""
    image_path: str = ""
    bbox_union_norm: Optional[NormalizedBBox] = None
    coord_system: str = "top-left"
    page_width_pts: float = 0.0
    page_height_pts: float = 0.0
    bbox_union_abs: Optional[Dict[str, float]] = None
    source_element_ids: List[int] = field(default_factory=list)
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    order_rank: int = 0   # stable reading-order index within the doc
    # Section-aware fields (backward compatible defaults)
    section_id: str = ""
    section_path: str = ""     # e.g., "3 Risk Overview > 3.2 Financial Risks"
    section_level: int = 0
    section_index: int = 0     # order within its section
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    content_sha256: str = ""

    def finalize(self) -> None:
        h = hashlib.sha256()
        h.update(self.text.encode("utf-8"))
        if self.html:
            h.update(b"\x00")
            h.update(self.html.encode("utf-8"))
        self.content_sha256 = h.hexdigest()

# ---- Document-level manifest --------------------------------------------------
@dataclass
class Manifest:
    doc_id: str
    source_file: str
    pages: Dict[int, Dict[str, float]]    # {page: {"width": w, "height": h}}
    coord_system: Literal["normalized_top_left"] = "normalized_top_left"
    notes: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

# ---- Serialization helpers ----------------------------------------------------
def to_jsonl(path: str, records: List[object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            obj = asdict(r)
            # bbox objects are already converted to dicts by asdict()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
