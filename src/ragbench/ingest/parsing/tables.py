#!/usr/bin/env python3
"""
Table extraction utilities for PDF processing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False

from .utils import get_file_metadata

logger = logging.getLogger(__name__)


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
