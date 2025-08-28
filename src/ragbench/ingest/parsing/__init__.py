"""
PDF parsing and ingestion package for RAG benchmarking.
"""

from .parse import process_pdf, main
from .processors import extract_element_data, parse_pdf_with_unstructured
from .tables import extract_tables_with_camelot
from .utils import normalize_text, find_pdf_files

__all__ = [
    'process_pdf',
    'main',
    'extract_element_data',
    'parse_pdf_with_unstructured',
    'extract_tables_with_camelot',
    'normalize_text',
    'find_pdf_files',
]
