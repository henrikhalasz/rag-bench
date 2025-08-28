

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
import pytest

# Ensure src/ is in sys.path for imports
SRC_PATH = str(Path(__file__).resolve().parents[3] / "src")
if SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)
from ragbench.ingest.parsing import utils

def test_normalize_text_basic():
	text = "This  is   a\n\nparagraph.\n\nAnother   one."
	expected = "This is a\n\nparagraph.\n\nAnother one."
	assert utils.normalize_text(text) == expected

def test_normalize_text_dehyphenate():
	text = "inter-\nnational"
	assert "international" in utils.normalize_text(text)

def test_normalize_text_unicode():
	text = "A\u00adB\xa0C"
	result = utils.normalize_text(text)
	assert "\u00ad" not in result
	assert "\xa0" not in result

def test_compute_doc_id_stable(tmp_path):
	file1 = tmp_path / "file1.txt"
	file1.write_text("abc")
	doc_id1 = utils.compute_doc_id(file1)
	doc_id2 = utils.compute_doc_id(file1)
	assert doc_id1 == doc_id2
	assert len(doc_id1) == 16

def test_get_file_metadata(tmp_path):
	file1 = tmp_path / "file1.txt"
	file1.write_text("abc")
	meta = utils.get_file_metadata(file1)
	assert "last_modified" in meta
	assert meta["languages"] == ["eng"]

def test_write_jsonl_atomic_and_read(tmp_path):
	output = tmp_path / "out.jsonl"
	records = [{"a": 1}, {"b": 2}]
	utils.write_jsonl_atomic(output, records)
	with open(output) as f:
		lines = [json.loads(line) for line in f]
	assert lines == records

def test_find_pdf_files(tmp_path):
	(tmp_path / "a.pdf").write_text("")
	(tmp_path / "b.txt").write_text("")
	sub = tmp_path / "sub"
	sub.mkdir()
	(sub / "c.pdf").write_text("")
	found = utils.find_pdf_files(tmp_path)
	found_names = {f.name for f in found}
	assert "a.pdf" in found_names
	assert "c.pdf" in found_names
	assert "b.txt" not in found_names
