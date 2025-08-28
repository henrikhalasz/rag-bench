
import sys
from pathlib import Path
import tempfile
import pytest
from unittest import mock

# Ensure src/ is in sys.path for imports
SRC_PATH = str(Path(__file__).resolve().parents[3] / "src")
if SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)
from ragbench.ingest.parsing import tables

def test_extract_tables_with_camelot_no_camelot(tmp_path, monkeypatch):
	monkeypatch.setattr(tables, "HAS_CAMELOT", False)
	pdf_path = tmp_path / "file.pdf"
	pdf_path.write_text("")
	out = tables.extract_tables_with_camelot(pdf_path, tmp_path, "docid")
	assert out == []

@mock.patch("ragbench.ingest.parsing.tables.camelot")
def test_extract_tables_with_camelot_success(mock_camelot, tmp_path):
	class DummyTable:
		def __init__(self, page, df):
			self.page = page
			self.df = df
	class DummyDF:
		def to_markdown(self, index=False):
			return "|a|b|"
		def to_string(self, index=False):
			return "a b"
	dummy_table = DummyTable(page=1, df=DummyDF())
	mock_camelot.read_pdf.side_effect = [[dummy_table], []]
	pdf_path = tmp_path / "file.pdf"
	pdf_path.write_text("")
	out = tables.extract_tables_with_camelot(pdf_path, tmp_path, "docid")
	assert isinstance(out, list)
	assert out[0]["type"] == "Table"
	assert out[0]["markdown_path"].endswith(".md")
