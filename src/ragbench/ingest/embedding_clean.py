#!/usr/bin/env python3
"""
embedding.py — Local HF embeddings + FAISS with Figure/Table enrichment

What this script does
- Loads chunks from a JSONL file (as produced by your chunker)
- Applies an embed policy to choose which chunks to embed
- Produces a *rich text* for embedding per chunk:
    • paragraphs/headings → use `text`
    • tables → optional HTML linearization to preserve structure
    • figures → optional local VLM (LLaVA) captioning + original caption text
- Tokenizes + truncates (head/tail/head+tail)
- Embeds with a local Hugging Face model (default: BAAI/bge-large-en-v1.5)
- (Optional) L2-normalizes vectors → cosine similarity via FAISS IndexFlatIP
- Writes artifacts:
    1) FAISS index file (.faiss)
    2) Parquet file containing vectors + metadata (+ the exact text embedded)
- Optional on-disk embedding cache

Notes
- API adapters intentionally omitted (local-only path).
- Deterministic: input sorted; outputs reproducible; portable Parquet.
- VLM and table linearization are toggleable via flags.

Quickstart (fully local demo):
    python embedding.py \
      --make-demo \
      --chunks demo_chunks.jsonl \
      --out-faiss demo.index.faiss \
      --out-parquet demo.vectors.parquet \
      --model BAAI/bge-large-en-v1.5 \
      --batch-size 16 --l2-normalize

Enable table linearization and VLM figure captioning on your own data:
    python embedding.py --chunks <your_chunks.jsonl> --out-faiss index.faiss \
      --out-parquet vectors.parquet --enable-vlm --linearize-tables --l2-normalize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Third-party deps
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("FAISS is required. Install faiss-cpu or faiss-gpu.") from e

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise RuntimeError("pyarrow is required for Parquet output. pip install pyarrow") from e

# Optional deps for enrichment (graceful fallback if missing)
try:
    from bs4 import BeautifulSoup  # for table linearization
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    from PIL import Image  # for VLM image loading
except Exception:
    Image = None  # type: ignore

# ----------------------------- CLI / Config ----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed JSONL chunks into FAISS + Parquet (with figure/table enrichment)")
    p.add_argument("--chunks", type=Path, default=Path("demo_chunks.jsonl"), help="Input chunks JSONL")
    p.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON (not required)")
    p.add_argument("--model", type=str, default="BAAI/bge-large-en-v1.5", help="HF text embedding model id")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"], help="Device selection for text encoder")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per chunk text before truncation")
    p.add_argument("--truncate", type=str, default="head+tail", choices=["head","tail","head+tail"], help="Truncation strategy for long texts")
    p.add_argument("--l2-normalize", action="store_true", help="L2-normalize vectors (recommended for cosine)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float16"], help="Vector dtype in Parquet")
    p.add_argument("--policy", type=str, default="roles=title,h1,h2,h3,paragraph,list_item,caption,table,figure|min_len=15|skip_empty_figures", help="Embed policy string")
    p.add_argument("--cache-dir", type=Path, default=None, help="Optional on-disk embeddings cache directory")
    p.add_argument("--out-faiss", type=Path, default=Path("vectors.index.faiss"), help="Output FAISS index path")
    p.add_argument("--out-parquet", type=Path, default=Path("vectors.parquet"), help="Output Parquet path (vectors + metadata)")
    p.add_argument("--make-demo", action="store_true", help="Create a demo_chunks.jsonl if it doesn't exist, then proceed")
    # enrichment toggles
    p.add_argument("--linearize-tables", action="store_true", help="Linearize table HTML into descriptive text before embedding")
    p.add_argument("--enable-vlm", action="store_true", help="Use a local VLM (LLaVA) to summarize figures (role=figure)")
    p.add_argument("--vlm-model-id", type=str, default="llava-hf/llava-1.5-7b-hf", help="LLaVA (vision-language) model id for local figure captioning")
    return p.parse_args()

# ----------------------------- Utilities -------------------------------------

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    role: str
    text: str
    page_start: int = 0
    order_rank: int = 0
    section_id: str = ""
    section_path: str = ""
    content_sha256: str = ""
    # Neighbor chain fields (for DataStore compatibility)
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    # Enrichment fields
    image_path: str = ""  # for figures
    html: str = ""        # for tables

# ----------------------------- Demo dataset ----------------------------------

def maybe_make_demo(path: Path) -> None:
    if not path.exists():
        rows = [
            {
                "chunk_id": "c0", "doc_id": "demo-doc", "role": "title",
                "text": "3 Risk Overview", "page_start": 1, "order_rank": 0,
                "section_id": "s0", "section_path": "3 Risk Overview",
                "content_sha256": hashlib.sha256(b"3 Risk Overview").hexdigest(),
            },
            {
                "chunk_id": "c1", "doc_id": "demo-doc", "role": "paragraph",
                "text": "This section summarizes key risk categories across the program.",
                "page_start": 1, "order_rank": 1, "section_id": "s0",
                "section_path": "3 Risk Overview",
                "content_sha256": hashlib.sha256(b"This section summarizes...").hexdigest(),
            },
            {
                "chunk_id": "c2", "doc_id": "demo-doc", "role": "h2",
                "text": "3.2 Financial Risks", "page_start": 1, "order_rank": 2,
                "section_id": "s1", "section_path": "3 Risk Overview > 3.2 Financial Risks",
                "content_sha256": hashlib.sha256(b"3.2 Financial Risks").hexdigest(),
            },
            {
                "chunk_id": "c3", "doc_id": "demo-doc", "role": "paragraph",
                "text": "Liquidity constraints and budget overruns impact delivery timelines.",
                "page_start": 1, "order_rank": 3, "section_id": "s1",
                "section_path": "3 Risk Overview > 3.2 Financial Risks",
                "content_sha256": hashlib.sha256(b"Liquidity constraints...").hexdigest(),
            },
            # Example figure (no real image in demo)
            {
                "chunk_id": "c4", "doc_id": "demo-doc", "role": "figure",
                "text": "Figure: Budget vs. Burn-down chart", "page_start": 2, "order_rank": 4,
                "section_id": "s1", "section_path": "3 Risk Overview > 3.2 Financial Risks",
                "image_path": "demo_images/figure-1.png",  # may not exist; VLM will skip
                "content_sha256": hashlib.sha256(b"Figure caption").hexdigest(),
            },
            # Example table (simple HTML)
            {
                "chunk_id": "c5", "doc_id": "demo-doc", "role": "table",
                "text": "Financial KPIs",
                "html": "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody><tr><td>Runway</td><td>9 mo</td></tr><tr><td>Burn</td><td>$120k/mo</td></tr></tbody></table>",
                "page_start": 2, "order_rank": 5, "section_id": "s1",
                "section_path": "3 Risk Overview > 3.2 Financial Risks",
                "content_sha256": hashlib.sha256(b"table html").hexdigest(),
            },
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[demo] Wrote {path}")

# ----------------------------- Policy ----------------------------------------

def parse_policy(policy: str) -> Dict[str, object]:
    # Example: "roles=title,h1,h2,h3,paragraph,list_item,caption,table,figure|min_len=15|skip_empty_figures"
    parts = [p.strip() for p in policy.split("|") if p.strip()]
    roles: List[str] = []
    min_len = 0
    skip_empty_figures = False
    for p in parts:
        if p.startswith("roles="):
            roles = [r.strip() for r in p[len("roles="):].split(",") if r.strip()]
        elif p.startswith("min_len="):
            try:
                min_len = int(p.split("=", 1)[1])
            except Exception:
                min_len = 0
        elif p == "skip_empty_figures":
            skip_empty_figures = True
    return {"roles": roles, "min_len": min_len, "skip_empty_figures": skip_empty_figures}

def select_chunks(rows: List["ChunkRow"], policy: Dict[str, object]) -> List["ChunkRow"]:
    roles = set(policy.get("roles", []))
    min_len = int(policy.get("min_len", 0))
    skip_empty_figures = bool(policy.get("skip_empty_figures", False))

    out: List[ChunkRow] = []
    for r in rows:
        if roles and r.role not in roles:
            if not (r.role == "figure" and not skip_empty_figures):
                continue
        if r.role == "figure" and skip_empty_figures and (not r.text or not r.text.strip()):
            continue
        if len((r.text or "").strip()) < min_len and r.role != "table":
            # allow tables with short captions if HTML present
            if not (r.role == "table" and r.html):
                continue
        out.append(r)
    return out

# ----------------------------- Neighbor Chain Rewiring -----------------------

def rewire_neighbor_chains(all_chunks: List[ChunkRow], kept_chunks: List[ChunkRow]) -> List[ChunkRow]:
    """Rewire neighbor chains so that prev/next point to the nearest *kept* chunk within each doc.

    Complexity: O(N) total (two linear passes per document). Stable and deterministic.
    Preconditions:
      - all_chunks are in original document order
      - kept_chunks are a filtered subset of all_chunks (any order)
    Postconditions:
      - Returned list contains only kept chunks, in original order
      - prev_id/next_id refer only to kept chunks (or None)
    """
    if not kept_chunks:
        return []

    # 0) Build fast lookups
    kept_ids = {c.chunk_id for c in kept_chunks}
    pos: Dict[str, int] = {c.chunk_id: i for i, c in enumerate(all_chunks)}  # absolute order

    # 1) Group indices by doc_id (preserves original order)
    by_doc: Dict[str, List[int]] = {}
    for i, c in enumerate(all_chunks):
        by_doc.setdefault(c.doc_id, []).append(i)

    # 2) Compute new links for kept chunks
    new_links: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    for doc_id, idxs in by_doc.items():
        # Kept mask inside this document
        kept_mask = [all_chunks[i].chunk_id in kept_ids for i in idxs]
        if not any(kept_mask):
            continue

        # Forward pass: nearest kept strictly BEFORE j
        prev_kept_idx: List[Optional[int]] = [None] * len(idxs)
        last_kept: Optional[int] = None
        for j in range(len(idxs)):
            prev_kept_idx[j] = last_kept
            if kept_mask[j]:
                last_kept = j

        # Backward pass: nearest kept strictly AFTER j
        next_kept_idx: List[Optional[int]] = [None] * len(idxs)
        next_kept: Optional[int] = None
        for j in range(len(idxs) - 1, -1, -1):
            if kept_mask[j]:
                # next_kept for THIS j must be a kept to the right, so don't set to self
                next_kept_idx[j] = next_kept
                next_kept = j
            else:
                next_kept_idx[j] = next_kept

        # Assign links for kept rows only
        for j, i_abs in enumerate(idxs):
            if not kept_mask[j]:
                continue
            cid = all_chunks[i_abs].chunk_id
            pk = prev_kept_idx[j]
            nk = next_kept_idx[j]
            prev_id = all_chunks[idxs[pk]].chunk_id if pk is not None else None
            next_id = all_chunks[idxs[nk]].chunk_id if nk is not None else None
            new_links[cid] = (prev_id, next_id)

    # 3) Build final kept list in ORIGINAL order; set rewired ids
    kept_sorted = sorted(kept_chunks, key=lambda r: pos[r.chunk_id])
    rewired: List[ChunkRow] = []
    for c in kept_sorted:
        prev_id, next_id = new_links.get(c.chunk_id, (None, None))
        # Prefer dataclass.replace so fields remain complete and type-safe
        rewired.append(replace(c, prev_id=prev_id, next_id=next_id))

    # 4) Integrity checks
    for c in rewired:
        if c.prev_id is not None and c.prev_id not in kept_ids:
            raise ValueError(f"Dangling prev_id after rewiring: {c.chunk_id} -> {c.prev_id}")
        if c.next_id is not None and c.next_id not in kept_ids:
            raise ValueError(f"Dangling next_id after rewiring: {c.chunk_id} -> {c.next_id}")
        if c.prev_id == c.chunk_id or c.next_id == c.chunk_id:
            raise ValueError(f"Self-link detected after rewiring: {c.chunk_id}")

    return rewired

# ----------------------------- IO --------------------------------------------

def load_chunks_jsonl(path: Path) -> List[ChunkRow]:
    base_dir = path.parent.resolve()
    rows: List[ChunkRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            img = d.get("image_path", "") or ""
            # try to resolve relative to chunks file directory
            if img and not Path(img).exists():
                p_try = base_dir / img
                if p_try.exists():
                    img = str(p_try)
            rows.append(ChunkRow(
                chunk_id=d.get("chunk_id"),
                doc_id=d.get("doc_id", ""),
                role=d.get("role", "paragraph"),
                text=d.get("text", ""),
                page_start=int(d.get("page_start", 0)),
                order_rank=int(d.get("order_rank", 0)),
                section_id=d.get("section_id", ""),
                section_path=d.get("section_path", ""),
                content_sha256=d.get("content_sha256", ""),
                prev_id=d.get("prev_id"),
                next_id=d.get("next_id"),
                image_path=img,
                html=d.get("html", ""),
            ))
    # Deterministic order
    rows.sort(key=lambda r: (r.doc_id, r.page_start, r.order_rank, r.chunk_id))
    return rows

# ----------------------------- Tokenization / Truncation ---------------------

def count_tokens(tok: AutoTokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

@dataclass
class TruncationSpec:
    strategy: str
    max_tokens: int

    def signature(self) -> str:
        return f"{self.strategy}({self.max_tokens})"

def apply_truncation(tok: AutoTokenizer, text: str, spec: TruncationSpec) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= spec.max_tokens:
        return text
    if spec.strategy == "head":
        ids2 = ids[:spec.max_tokens]
    elif spec.strategy == "tail":
        ids2 = ids[-spec.max_tokens:]
    else:  # head+tail
        head = int(spec.max_tokens * 0.7)
        tail = spec.max_tokens - head
        ids2 = ids[:head] + ids[-tail:]
    return tok.decode(ids2, skip_special_tokens=True)

# ----------------------------- HF Model Adapter ------------------------------

@dataclass
class HFAdapter:
    model_id: str
    device: torch.device
    tokenizer: AutoTokenizer
    model: AutoModel
    max_model_len: int

    @classmethod
    def load(cls, model_id: str, device_pref: str = "auto") -> "HFAdapter":
        if device_pref == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_pref)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        max_len = int(getattr(tokenizer, "model_max_length", 8192) or 8192)
        return cls(model_id=model_id, device=device, tokenizer=tokenizer, model=model, max_model_len=max_len)

    def embed(self, texts: List[str], batch_size: int = 32, l2_normalize: bool = True) -> np.ndarray:
        out_vecs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                # Mean pooling with attention mask
                last_hidden = outputs.last_hidden_state  # [B, T, H]
                mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                masked = last_hidden * mask
                sum_emb = masked.sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-6)
                mean_emb = sum_emb / lengths
                vecs = mean_emb
                if l2_normalize:
                    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                out_vecs.append(vecs.detach().cpu().numpy())
        return np.concatenate(out_vecs, axis=0) if out_vecs else np.zeros((0, self.dim()))

    def dim(self) -> int:
        return int(self.model.config.hidden_size)

# ----------------------------- Caching ---------------------------------------

def cache_key(model_id: str, schema_version: str, text_sha256: str, truncation_sig: str) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b" ")
    h.update(schema_version.encode("utf-8"))
    h.update(b" ")
    h.update(text_sha256.encode("utf-8"))
    h.update(b" ")
    h.update(truncation_sig.encode("utf-8"))
    return h.hexdigest()

def cache_get(cache_dir: Optional[Path], key: str) -> Optional[np.ndarray]:
    if not cache_dir:
        return None
    path = cache_dir / f"{key}.npy"
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            return None
    return None

def cache_put(cache_dir: Optional[Path], key: str, vec: np.ndarray) -> None:
    if not cache_dir:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npy"
    try:
        np.save(path, vec)
    except Exception:
        pass

# ----------------------------- FAISS helpers ---------------------------------

def build_faiss_index(vectors: np.ndarray, normalize: bool = True) -> faiss.Index:
    if normalize:
        index = faiss.IndexFlatIP(vectors.shape[1])
    else:
        index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index

# ----------------------------- Parquet writer --------------------------------

def write_parquet(
    out_path: Path,
    model_id: str,
    dim: int,
    trunc_sig: str,
    l2_normalize: bool,
    dtype: str,
    rows: List[ChunkRow],
    vectors: np.ndarray,
    embedding_texts: List[str],
) -> None:
    if dtype == "float16":
        vec_store = vectors.astype(np.float16)
    else:
        vec_store = vectors.astype(np.float32)
    vec_type = pa.list_(pa.float16()) if dtype == "float16" else pa.list_(pa.float32())

    table = pa.table({
        "chunk_id": pa.array([r.chunk_id for r in rows], pa.string()),
        "doc_id": pa.array([r.doc_id for r in rows], pa.string()),
        "role": pa.array([r.role for r in rows], pa.string()),
        "page_start": pa.array([r.page_start for r in rows], pa.int32()),
        "order_rank": pa.array([r.order_rank for r in rows], pa.int32()),
        "section_id": pa.array([r.section_id for r in rows], pa.string()),
        "section_path": pa.array([r.section_path for r in rows], pa.string()),
        "content_sha256": pa.array([r.content_sha256 for r in rows], pa.string()),
        "prev_id": pa.array([r.prev_id for r in rows], pa.string()),
        "next_id": pa.array([r.next_id for r in rows], pa.string()),
        "model_id": pa.array([model_id] * len(rows), pa.string()),
        "dim": pa.array([dim] * len(rows), pa.int32()),
        "truncation": pa.array([trunc_sig] * len(rows), pa.string()),
        "l2_normalize": pa.array([l2_normalize] * len(rows), pa.bool_()),
        "created_at": pa.array([utc_now()] * len(rows), pa.string()),
        "embedding_text": pa.array(embedding_texts, pa.string()),
        "embedding_text_sha256": pa.array([hashlib.sha256(t.encode('utf-8')).hexdigest() for t in embedding_texts], pa.string()),
        "vector": pa.array(vec_store.tolist(), type=vec_type),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

# ----------------------------- Table linearization ---------------------------

def linearize_table_html(html_str: str, caption: str) -> str:
    """Converts table HTML into a descriptive, row-wise text. Fallbacks gracefully."""
    if not html_str or BeautifulSoup is None:
        return caption or ""
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table')
        if table is None:
            return caption or ""
        # headers
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        if not headers:
            # try the first row as header
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['td','th'])]
        # rows
        body = table.find('tbody') or table
        rows = body.find_all('tr') if body else []
        linearized_rows: List[str] = []
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all('td')]
            if not cells:
                continue
            pairs = []
            for i, cell in enumerate(cells):
                col = headers[i] if i < len(headers) else f"col{i+1}"
                if cell:
                    pairs.append(f"{col} is '{cell}'")
            if pairs:
                linearized_rows.append("; ".join(pairs))
        out = []
        if caption:
            out.append(f"Table caption: {caption}")
        if headers:
            out.append("Columns: " + ", ".join(headers))
        for i, row_str in enumerate(linearized_rows, start=1):
            out.append(f"Row {i}: {row_str}")
        return "\n".join(out).strip()
    except Exception:
        return caption or ""

# ----------------------------- VLM (LLaVA) summarizer ------------------------
# We memoize and load only if --enable-vlm and a figure is present.
_VLM_MODEL = None
_VLM_PROC = None


def _pick_vlm_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_figure_summary(image_path: str, vlm_model_id: str) -> str:
    """Generate a short description of a figure using a local LLaVA model.
    Falls back to empty string if deps/devices are not suitable."""
    global _VLM_MODEL, _VLM_PROC
    if not image_path:
        return ""
    p = Path(image_path)
    if not p.exists() or Image is None:
        return ""
    try:
        if _VLM_MODEL is None:
            # Import here to keep base path lightweight if VLM is disabled
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration  # type: ignore
            _VLM_PROC = LlavaNextProcessor.from_pretrained(vlm_model_id)
            device = _pick_vlm_device()
            # bitsandbytes 4-bit works only on CUDA; on MPS/CPU, use fp16/fp32
            use_4bit = (device.type == "cuda")
            if use_4bit:
                _VLM_MODEL = LlavaNextForConditionalGeneration.from_pretrained(
                    vlm_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True
                )
            else:
                dtype = torch.float16 if device.type == "mps" else torch.float32
                _VLM_MODEL = LlavaNextForConditionalGeneration.from_pretrained(
                    vlm_model_id, torch_dtype=dtype, low_cpu_mem_usage=True
                )
            _VLM_MODEL.to(device)
            _VLM_MODEL.eval()
        prompt = (
            "USER: <image>"
            "Describe the key information in this figure. If it is a chart, summarize axes, trends, outliers, and legend. "
            "Be concise but specific."
            "ASSISTANT:"
        )
        image = Image.open(str(p)).convert("RGB")
        device = next(_VLM_MODEL.parameters()).device
        inputs = _VLM_PROC(prompt, image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = _VLM_MODEL.generate(**inputs, max_new_tokens=192)
        text = _VLM_PROC.decode(output[0], skip_special_tokens=True)
        # Keep only the assistant response if the template leaks in
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        return text.strip()
    except Exception as e:
        print(f"[warn] VLM failed for {image_path}: {e}")
        return ""

# ----------------------------- Main flow -------------------------------------

def doc_prefix_for_bge(text: str) -> str:
    # BGE v1.5 recommends prefixing 'passage: ' for docs and 'query: ' for queries
    t = text or ""
    return f"passage: {t}" if not t.lower().startswith("passage:") else t


def write_vectors_and_index(
    out_parquet: Path,
    out_faiss: Path,
    adapter: HFAdapter,
    trunc: TruncationSpec,
    policy: Dict[str, object],
    rows_all: List[ChunkRow],
    args: argparse.Namespace,
) -> int:
    # Select rows per policy
    rows = select_chunks(rows_all, policy)
    if not rows:
        print("No chunks selected by policy.")
        return 1

    # Rewire neighbor chains to maintain consistency among kept chunks
    rows = rewire_neighbor_chains(rows_all, rows)
    print(f"[ok] rewired neighbor chains for {len(rows)} kept chunks")

    # 1) Build rich text per chunk (tables/figures)
    rich_texts: List[str] = []
    if args.linearize_tables and BeautifulSoup is None:
        print("[warn] --linearize-tables requested but beautifulsoup4 not installed; skipping linearization")
    gen_vlm = args.enable_vlm
    if gen_vlm and Image is None:
        print("[warn] --enable-vlm requested but Pillow not installed; skipping VLM")
        gen_vlm = False

    for r in rows:
        enriched = r.text or ""
        if args.linearize_tables and r.role == "table" and r.html:
            enriched = linearize_table_html(r.html, r.text)
        elif gen_vlm and r.role == "figure":
            # Combine original caption + VLM description (if any)
            vlm_txt = get_figure_summary(r.image_path, args.vlm_model_id)
            prefix = (r.text or "").strip()
            if vlm_txt:
                enriched = (f"Figure caption: {prefix}\nVisual summary: {vlm_txt}").strip()
            else:
                enriched = (f"Figure caption: {prefix}").strip()
        rich_texts.append(enriched)

    # 2) Cache-aware embedding pass
    adapter_dim = adapter.dim()
    vectors: List[np.ndarray] = []
    used_rows: List[ChunkRow] = []
    used_texts: List[str] = []

    schema_version = "chunk-v1"  # bump if you change upstream chunk format contract
    for r, rich in zip(rows, rich_texts):
        emb_text = doc_prefix_for_bge(rich)
        text_sha = hashlib.sha256(emb_text.encode("utf-8")).hexdigest()
        key = cache_key(adapter.model_id, schema_version, text_sha, trunc.signature())
        cached = cache_get(args.cache_dir, key)
        if cached is not None:
            vectors.append(cached)
            used_rows.append(r)
            used_texts.append(emb_text)
            continue

        # Truncate then embed
        txt = apply_truncation(adapter.tokenizer, emb_text, trunc)
        emb = adapter.embed([txt], batch_size=args.batch_size, l2_normalize=args.l2_normalize)
        if emb.shape != (1, adapter_dim):
            print(f"Unexpected embedding shape for {r.chunk_id}: {emb.shape}")
            return 1
        v = emb[0]
        if not np.isfinite(v).all():
            print(f"Non-finite vector for {r.chunk_id}")
            return 1
        cache_put(args.cache_dir, key, v)
        vectors.append(v)
        used_rows.append(r)
        used_texts.append(txt)

    V = np.stack(vectors, axis=0) if vectors else np.zeros((0, adapter_dim))

    # 3) Write Parquet (+ vectors + the *exact* embedded text)
    write_parquet(out_parquet, adapter.model_id, adapter_dim, trunc.signature(), args.l2_normalize, args.dtype, used_rows, V, used_texts)
    print(f"[ok] wrote Parquet → {out_parquet}")

    # 4) Build + write FAISS
    index = build_faiss_index(V, normalize=args.l2_normalize)
    faiss.write_index(index, str(out_faiss))
    print(f"[ok] wrote FAISS → {out_faiss} (ntotal={index.ntotal}, dim={V.shape[1] if V.size else adapter_dim})")

    return 0


# ----------------------------- Main ------------------------------------------

def main() -> int:
    args = parse_args()
    if args.make_demo:
        maybe_make_demo(args.chunks)

    # Load chunks (deterministic sort inside)
    rows_all = load_chunks_jsonl(args.chunks)
    if not rows_all:
        print("No rows found in chunks file.")
        return 1

    # HF text encoder
    adapter = HFAdapter.load(args.model, args.device)
    trunc = TruncationSpec(strategy=args.truncate, max_tokens=min(args.max_tokens, adapter.max_model_len))
    policy = parse_policy(args.policy)

    # Execute
    rc = write_vectors_and_index(
        out_parquet=args.out_parquet,
        out_faiss=args.out_faiss,
        adapter=adapter,
        trunc=trunc,
        policy=policy,
        rows_all=rows_all,
        args=args,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
