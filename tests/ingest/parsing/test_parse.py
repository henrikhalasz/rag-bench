import sys
from pathlib import Path
import tempfile
import pytest
from unittest import mock

# Ensure src/ is in sys.path for imports
SRC_PATH = str(Path(__file__).resolve().parents[3] / "src")
if SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)
from ragbench.ingest.parsing import parse

def test_process_pdf_success(monkeypatch, tmp_path):
	# Patch parse_pdf_with_unstructured to return dummy elements
	class DummyElement:
		def __init__(self, t):
			self.text = t
			self.metadata = type("Meta", (), {"page_number": 1, "text_as_html": "", "image_path": None})()
			self.text_as_html = ""
		def to_dict(self):
			return {"type": "Text"}
	monkeypatch.setattr(parse, "parse_pdf_with_unstructured", lambda *a, **kw: [DummyElement("foo")])
	monkeypatch.setattr(parse, "extract_element_data", lambda *a, **kw: {
		"doc_id": "docid", "type": "Text", "page_number": 1, "html": "", "image_path": "", "coordinates": None, "element_index": 0, "meta": {"languages": ["eng"]}, "text": "foo", "source_path": "", "file_name": ""
	})
	monkeypatch.setattr(parse, "write_jsonl_atomic", lambda *a, **kw: None)
	monkeypatch.setattr(parse, "compute_doc_id", lambda *a, **kw: "docid")
	pdf_path = tmp_path / "file.pdf"
	pdf_path.write_text("")
	out = parse.process_pdf(pdf_path, tmp_path)
	assert out["success"] is True
	assert out["elements"] == 1

@mock.patch("ragbench.ingest.parsing.parse.find_pdf_files")
@mock.patch("ragbench.ingest.parsing.parse.process_pdf")
def test_main_success(mock_process_pdf, mock_find_pdf_files, monkeypatch, tmp_path):
	mock_find_pdf_files.return_value = [tmp_path / "file.pdf"]
	mock_process_pdf.return_value = {"success": True, "elements": 1, "pages": 1, "tables": 0, "images": 0, "file": "file.pdf"}
	monkeypatch.setattr(parse, "HAS_UNSTRUCTURED", True)
	monkeypatch.setattr(parse, "HAS_CAMELOT", False)
	monkeypatch.setattr(parse, "setup_logging", lambda *a, **kw: None)
	args = ["--src", str(tmp_path), "--out", str(tmp_path), "--workers", "1"]
	with mock.patch("sys.argv", ["prog"] + args):
		assert parse.main() == 0

# Move the mock_process_pdf function to the top level so it can be pickled
def mock_process_pdf(pdf_path, output_dir, use_hires, table_fallback):
    return {
        "file": pdf_path.name,
        "success": True,
        "elements": 1,
        "pages": 1,
        "tables": 0,
        "images": 0
    }


def test_progress_bar(monkeypatch, tmp_path):
    """Test that the progress bar updates correctly."""
    monkeypatch.setattr(parse, "process_pdf", mock_process_pdf)

    # Create dummy PDF files
    pdf_files = [tmp_path / f"file_{i}.pdf" for i in range(3)]
    for pdf_file in pdf_files:
        pdf_file.write_text("dummy content")

    # Mock find_pdf_files to return the dummy files
    monkeypatch.setattr(parse, "find_pdf_files", lambda *args, **kwargs: pdf_files)

    # Mock setup_logging to avoid actual logging
    monkeypatch.setattr(parse, "setup_logging", lambda *args, **kwargs: None)

    # Mock HAS_UNSTRUCTURED to True
    monkeypatch.setattr(parse, "HAS_UNSTRUCTURED", True)

    # Mock tqdm to track progress bar updates
    with mock.patch("ragbench.ingest.parsing.parse.tqdm", wraps=parse.tqdm) as mock_tqdm:
        args = ["--src", str(tmp_path), "--out", str(tmp_path), "--workers", "2"]
        with mock.patch("sys.argv", ["prog"] + args):
            result = parse.main()

        # Ensure the progress bar was updated
        assert mock_tqdm.called

    assert result == 0
