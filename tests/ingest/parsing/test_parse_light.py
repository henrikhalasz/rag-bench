
import sys
from pathlib import Path
import tempfile
import pytest
from unittest import mock

# Ensure src/ is in sys.path for imports
SRC_PATH = str(Path(__file__).resolve().parents[3] / "src")
if SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)
from ragbench.ingest.parsing import parse_light

def test_process_pdf_success(monkeypatch, tmp_path):
	# Patch parse_pdf_with_unstructured to return dummy elements
	class DummyElement:
		def __init__(self, t):
			self.text = t
			self.metadata = type("Meta", (), {"page_number": 1, "text_as_html": "", "image_path": None})()
			self.text_as_html = ""
		def to_dict(self):
			return {"type": "Text"}
	monkeypatch.setattr(parse_light, "parse_pdf_with_unstructured", lambda *a, **kw: [DummyElement("foo")])
	monkeypatch.setattr(parse_light, "extract_element_data", lambda *a, **kw: {
		"doc_id": "docid", "type": "Text", "page_number": 1, "html": "", "image_path": "", "coordinates": None, "element_index": 0, "meta": {"languages": ["eng"]}, "text": "foo", "source_path": "", "file_name": ""
	})
	monkeypatch.setattr(parse_light, "write_jsonl_atomic", lambda *a, **kw: None)
	monkeypatch.setattr(parse_light, "compute_doc_id", lambda *a, **kw: "docid")
	pdf_path = tmp_path / "file.pdf"
	pdf_path.write_text("")
	out = parse_light.process_pdf(pdf_path, tmp_path)
	assert out["success"] is True
	assert out["elements"] == 1

@mock.patch("ragbench.ingest.parsing.parse_light.find_pdf_files")
@mock.patch("ragbench.ingest.parsing.parse_light.process_pdf")
def test_main_success(mock_process_pdf, mock_find_pdf_files, monkeypatch, tmp_path):
	mock_find_pdf_files.return_value = [tmp_path / "file.pdf"]
	mock_process_pdf.return_value = {"success": True, "elements": 1, "pages": 1, "tables": 0, "images": 0, "file": "file.pdf"}
	monkeypatch.setattr(parse_light, "HAS_UNSTRUCTURED", True)
	monkeypatch.setattr(parse_light, "HAS_CAMELOT", False)
	monkeypatch.setattr(parse_light, "setup_logging", lambda *a, **kw: None)
	args = ["--src", str(tmp_path), "--out", str(tmp_path)]
	with mock.patch("sys.argv", ["prog"] + args):
		assert parse_light.main() == 0
