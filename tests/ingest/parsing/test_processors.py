

import sys
from pathlib import Path
import tempfile
import pytest
from unittest import mock

# Ensure src/ is in sys.path for imports
SRC_PATH = str(Path(__file__).resolve().parents[3] / "src")
if SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)
from ragbench.ingest.parsing import processors

class DummyElement:
	def __init__(self, text="foo", meta=None):
		self.text = text
		self.metadata = meta
		self.text_as_html = "<table>...</table>"
	def to_dict(self):
		return {"type": "Table"}

class DummyMeta:
	def __init__(self):
		self.text_as_html = "<table>...</table>"
		self.page_number = 1
		self.image_path = None
		self.coordinates = None

def test_extract_element_data_basic(tmp_path):
	meta = DummyMeta()
	element = DummyElement("Some text", meta)
	doc_id = "docid"
	source_path = tmp_path / "file.pdf"
	source_path.write_text("dummy")
	image_dir = tmp_path / "images"
	result = processors.extract_element_data(element, doc_id, source_path, image_dir, 0)
	assert result["doc_id"] == doc_id
	assert result["type"] == "Table"
	assert result["text"] == "Some text"
	assert result["html"] == "<table>...</table>"
	assert result["meta"]["languages"] == ["eng"]

def test_extract_element_data_with_coordinates(tmp_path):
	class DummyCoords:
		def to_dict(self):
			return {"system": "PixelSpace", "layout_width": 100, "layout_height": 200, "points": [(0,0),(0,1),(2,3),(1,0)]}
	meta = DummyMeta()
	meta.coordinates = DummyCoords()
	element = DummyElement("Text", meta)
	doc_id = "docid"
	source_path = tmp_path / "file.pdf"
	source_path.write_text("dummy")
	image_dir = tmp_path / "images"
	result = processors.extract_element_data(element, doc_id, source_path, image_dir, 1)
	assert result["coordinates"] == {'x1': 0.0, 'y1': 0.0, 'x2': 2.0, 'y2': 3.0}
	assert result["meta"]["coord_system"] == "PixelSpace"
	assert result["meta"]["layout_size"] == {"w": 100.0, "h": 200.0}
	assert "bbox_norm_top_left" in result["meta"]

@mock.patch("ragbench.ingest.parsing.processors.partition_pdf")
def test_parse_pdf_with_unstructured_success(mock_partition_pdf, tmp_path):
	mock_partition_pdf.return_value = [DummyElement("foo")]
	pdf_path = tmp_path / "file.pdf"
	pdf_path.write_text("dummy")
	image_dir = tmp_path / "images"
	result = processors.parse_pdf_with_unstructured(pdf_path, image_dir, use_hires=True)
	assert isinstance(result, list)
	assert result[0].text == "foo"

@mock.patch("ragbench.ingest.parsing.processors.partition_pdf")
def test_parse_pdf_with_unstructured_fallback(mock_partition_pdf, tmp_path):
	def side_effect(**kwargs):
		if kwargs.get("strategy") == "hi_res":
			raise Exception("fail")
		return [DummyElement("bar")]
	mock_partition_pdf.side_effect = side_effect
	pdf_path = tmp_path / "file.pdf"
	pdf_path.write_text("dummy")
	image_dir = tmp_path / "images"
	result = processors.parse_pdf_with_unstructured(pdf_path, image_dir, use_hires=True)
	assert result[0].text == "bar"
