#!/usr/bin/env python3
"""
Utility functions for PDF processing and text normalization.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

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
    except ImportError:
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
