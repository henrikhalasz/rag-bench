
#!/usr/bin/env python3
"""
RAG Parser Quality Evaluator
----------------------------
Given a PDF and a JSONL of parsed elements that include coordinates,
this tool computes quality metrics and generates a visual HTML report.

Outputs
- report/index.html  : interactive report with side-by-side PDF page renders and overlay boxes
- report/images/     : page_{i}.png and page_{i}_overlay.png
- report/metrics.json: global + per-page metrics (JSON)
- report/metrics.csv : same metrics in CSV form

Usage
  python rag_parser_quality_eval.py --pdf path/to.pdf --jsonl path/to.jsonl --outdir report_dir [--max-pages N]
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import math
import re

# Rendering
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# Data
import pandas as pd


@dataclass
class Element:
    doc_id: str
    page_number: int
    type: str
    text: str
    html: str
    image_path: str
    coordinates: Dict[str, float]
    element_index: int
    meta: Dict[str, Any]


def load_jsonl(path: Path) -> List[Element]:
    rows: List[Element] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(Element(
                doc_id=obj.get("doc_id", ""),
                page_number=int(obj.get("page_number", 1)),
                type=obj.get("type", "Unknown"),
                text=obj.get("text", ""),
                html=obj.get("html", ""),
                image_path=obj.get("image_path", ""),
                coordinates=obj.get("coordinates", {}) or {},
                element_index=int(obj.get("element_index", -1)),
                meta=obj.get("meta", {}) or {},
            ))
    return rows


def rasterize_page(doc: fitz.Document, page_num: int, zoom: float = 2.0) -> Image.Image:
    """
    Render a PDF page to a PIL image.
    """
    page = doc.load_page(page_num - 1)  # 1-indexed input
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def draw_overlays(page_img: Image.Image, elements: List[Element], page_meta_size: Tuple[int,int]) -> Image.Image:
    """
    Draw bounding boxes for elements on a copy of the page image.
    We assume element.meta['layout_size'] provides page width/height the coordinates are defined in.
    If not present, we fallback to the rendered image size.
    """
    overlay = page_img.copy()
    draw = ImageDraw.Draw(overlay)

    # Try to load a default font; PIL always has one
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    img_w, img_h = overlay.size
    meta_w, meta_h = page_meta_size

    x_scale = img_w / meta_w if meta_w else 1.0
    y_scale = img_h / meta_h if meta_h else 1.0

    for i, el in enumerate(elements):
        coords = el.coordinates or {}
        x1, y1 = coords.get("x1", 0), coords.get("y1", 0)
        x2, y2 = coords.get("x2", 0), coords.get("y2", 0)

        # Scale from meta space to image space
        bx1 = x1 * x_scale
        by1 = y1 * y_scale
        bx2 = x2 * x_scale
        by2 = y2 * y_scale

        # Draw rectangle (default color)
        draw.rectangle([bx1, by1, bx2, by2], outline=1, width=2)
        label = f"{el.element_index}:{el.type}"
        # Draw a small label background
        tw = draw.textlength(label, font=font) if font else (len(label) * 6)
        th = 12
        draw.rectangle([bx1, max(0, by1- (th+4)), bx1 + tw + 6, by1], fill=(240,240,240))
        if font:
            draw.text((bx1+3, max(0, by1-(th+3))), label, fill=(0,0,0), font=font)
        else:
            draw.text((bx1+3, max(0, by1-(th+3))), label, fill=(0,0,0))

    return overlay


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def text_recall_precision(pdf_text: str, parsed_text: str) -> Tuple[float, float, float]:
    """
    Compute simple token-based recall, precision, and F1 between PDF text and parsed text.
    """
    t_pdf = tokenize(pdf_text)
    t_par = tokenize(parsed_text)
    set_pdf, set_par = set(t_pdf), set(t_par)
    if not set_pdf and not set_par:
        return 1.0, 1.0, 1.0
    if not set_pdf:
        return 1.0, 0.0, 0.0
    if not set_par:
        return 0.0, 1.0, 0.0

    tp = len(set_pdf & set_par)
    recall = tp / len(set_pdf) if set_pdf else 0.0
    precision = tp / len(set_par) if set_par else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return recall, precision, f1


def page_images_count(page: fitz.Page) -> int:
    try:
        imgs = page.get_images(full=True)
        return len(imgs)
    except Exception:
        return 0


def bbox_area(x1, y1, x2, y2) -> float:
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def compute_layout_metrics(page_elems: List[Element], page_meta_size: Tuple[int,int]) -> Dict[str, Any]:
    W, H = page_meta_size
    in_bounds = 0
    total = len(page_elems)
    total_area = 0.0
    overlap_area = 0.0

    boxes = []
    for el in page_elems:
        c = el.coordinates or {}
        x1, y1, x2, y2 = c.get("x1", 0), c.get("y1", 0), c.get("x2", 0), c.get("y2", 0)
        # in-bounds check (allow slight negative/overflow tolerance of 2px)
        if x1 >= -2 and y1 >= -2 and x2 <= W + 2 and y2 <= H + 2 and x2 > x1 and y2 > y1:
            in_bounds += 1
        area = bbox_area(x1, y1, x2, y2)
        total_area += area
        boxes.append((x1, y1, x2, y2))

    # Pairwise overlap (O(n^2) but pages are typically small)
    def inter(a,b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1); y1 = max(ay1, by1)
        x2 = min(ax2, bx2); y2 = min(ay2, by2)
        return bbox_area(x1,y1,x2,y2)

    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            overlap_area += inter(boxes[i], boxes[j])

    page_area = W * H if W and H else 0.0
    coverage = min(1.0, total_area / page_area) if page_area else 0.0
    overlap_ratio = overlap_area / page_area if page_area else 0.0

    return {
        "elements": total,
        "in_bounds_ratio": (in_bounds / total) if total else 1.0,
        "area_coverage_ratio": coverage,
        "overlap_area_ratio": overlap_ratio,
    }


def build_report(pdf_path: Path, jsonl_path: Path, outdir: Path, max_pages: int = None) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    img_dir = outdir / "images"
    img_dir.mkdir(exist_ok=True)

    # Load parsed elements
    elements = load_jsonl(jsonl_path)

    # Group elements by page
    by_page: Dict[int, List[Element]] = {}
    for el in elements:
        by_page.setdefault(el.page_number, []).append(el)

    # Open PDF
    doc = fitz.open(pdf_path.as_posix())

    # Compute per-page
    pages = range(1, doc.page_count + 1)
    if max_pages is not None:
        pages = range(1, min(doc.page_count, max_pages) + 1)

    per_page_metrics = []
    global_text_pdf = []
    global_text_parsed = []

    for p in pages:
        page = doc.load_page(p-1)
        page_text = page.get_text("text")

        # Determine page meta size from any element (assume consistent)
        elems = by_page.get(p, [])
        if elems and elems[0].meta and elems[0].meta.get("layout_size"):
            W = int(round(elems[0].meta["layout_size"]["w"]))
            H = int(round(elems[0].meta["layout_size"]["h"]))
        else:
            # fallback to PDF pixel size at 72dpi
            rect = page.rect
            W, H = int(rect.width), int(rect.height)

        # Render images
        page_img = rasterize_page(doc, p, zoom=2.0)
        overlay_img = draw_overlays(page_img, elems, (W, H))

        # Save images
        page_img_path = img_dir / f"page_{p:03d}.png"
        overlay_img_path = img_dir / f"page_{p:03d}_overlay.png"
        page_img.save(page_img_path.as_posix())
        overlay_img.save(overlay_img_path.as_posix())

        # Metrics
        parsed_text_page = " ".join([e.text for e in elems if e.text])
        rec, prec, f1 = text_recall_precision(page_text, parsed_text_page)
        layout = compute_layout_metrics(elems, (W, H))
        n_pdf_imgs = len(page.get_images(full=True))
        n_parsed_imgs = sum(1 for e in elems if (e.type.lower() in ["figure","image"]) or (e.image_path))

        per_page_metrics.append({
            "page": p,
            "elements": layout["elements"],
            "text_recall": rec,
            "text_precision": prec,
            "text_f1": f1,
            "in_bounds_ratio": layout["in_bounds_ratio"],
            "area_coverage_ratio": layout["area_coverage_ratio"],
            "overlap_area_ratio": layout["overlap_area_ratio"],
            "pdf_images_detected": n_pdf_imgs,
            "parsed_figures_reported": n_parsed_imgs,
            "page_image": page_img_path.name,
            "overlay_image": overlay_img_path.name,
        })

        global_text_pdf.append(page_text)
        global_text_parsed.append(parsed_text_page)

    # Global metrics
    global_recall, global_precision, global_f1 = text_recall_precision("\n".join(global_text_pdf),
                                                                       "\n".join(global_text_parsed))
    # Count elements by type
    type_counts = {}
    for e in elements:
        t = e.type or "Unknown"
        type_counts[t] = type_counts.get(t, 0) + 1

    # Ordering sanity checks per page
    ordering_issues = []
    for p, elems in by_page.items():
        idxs = [e.element_index for e in elems if e.element_index is not None]
        if idxs and sorted(idxs) != idxs:
            ordering_issues.append({"page": p, "issue": "element_index not non-decreasing"})

    summary = {
        "pdf": pdf_path.name,
        "jsonl": jsonl_path.name,
        "pages_analyzed": len(list(pages)),
        "global_text_recall": global_recall,
        "global_text_precision": global_precision,
        "global_text_f1": global_f1,
        "element_type_counts": type_counts,
        "ordering_issues": ordering_issues,
        "per_page": per_page_metrics,
    }

    # Write JSON + CSV
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df = pd.DataFrame(per_page_metrics)
    df.to_csv(outdir / "metrics.csv", index=False)

    # Build HTML report
    html = build_html_report(summary, outdir)
    (outdir / "index.html").write_text(html, encoding="utf-8")

    return summary


def build_html_report(summary: Dict[str, Any], outdir: Path) -> str:
    rows = []
    for m in summary["per_page"]:
        rows.append(f"""
        <tr>
          <td>{m['page']}</td>
          <td>{m['elements']}</td>
          <td>{m['text_recall']:.3f}</td>
          <td>{m['text_precision']:.3f}</td>
          <td>{m['text_f1']:.3f}</td>
          <td>{m['in_bounds_ratio']:.3f}</td>
          <td>{m['area_coverage_ratio']:.3f}</td>
          <td>{m['overlap_area_ratio']:.3f}</td>
          <td>{m['pdf_images_detected']}</td>
          <td>{m['parsed_figures_reported']}</td>
          <td>
            <figure style="margin:0">
              <img src="images/{m['page_image']}" alt="page {m['page']}" style="max-width:350px; display:block; margin-bottom:6px; border:1px solid #ddd" />
              <figcaption style="font-size:12px;color:#666">Rendered</figcaption>
            </figure>
          </td>
          <td>
            <figure style="margin:0">
              <img src="images/{m['overlay_image']}" alt="overlay {m['page']}" style="max-width:350px; display:block; margin-bottom:6px; border:1px solid #ddd" />
              <figcaption style="font-size:12px;color:#666">Overlay</figcaption>
            </figure>
          </td>
        </tr>
        """)

    type_counts = summary.get("element_type_counts", {})
    type_list = "".join(f"<li>{k}: {v}</li>" for k,v in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    ordering_issue_html = "<br/>".join(f"Page {i['page']}: {i['issue']}" for i in summary.get("ordering_issues", [])) or "None"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>RAG Parser Quality Report – {summary['pdf']}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
  h1,h2 {{ margin: 0.2em 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  .muted {{ color: #555; }}
  .small {{ font-size: 13px; }}
  .kpi {{ display:flex; gap:24px; flex-wrap:wrap; }}
  .kpi div {{ background:#f8f8f8; padding:12px 16px; border-radius:8px; }}
  ul {{ margin: 0; padding-left: 18px; }}
</style>
</head>
<body>
  <h1>RAG Parser Quality Report</h1>
  <div class="muted small">{summary['pdf']} &middot; JSONL: {summary['jsonl']} &middot; Pages analyzed: {summary['pages_analyzed']}</div>

  <div class="kpi" style="margin-top:12px">
    <div><b>Global Recall</b><br/>{summary['global_text_recall']:.3f}</div>
    <div><b>Global Precision</b><br/>{summary['global_text_precision']:.3f}</div>
    <div><b>Global F1</b><br/>{summary['global_text_f1']:.3f}</div>
  </div>

  <h2 style="margin-top:24px">Element Types</h2>
  <ul>{type_list}</ul>

  <h2 style="margin-top:24px">Ordering Issues</h2>
  <div class="small">{ordering_issue_html}</div>

  <h2 style="margin-top:24px">Per-Page Metrics & Visuals</h2>
  <table>
    <thead>
      <tr>
        <th>Page</th>
        <th>#Elems</th>
        <th>Recall</th>
        <th>Precision</th>
        <th>F1</th>
        <th>In-bounds</th>
        <th>Area Cover</th>
        <th>Overlap</th>
        <th>#PDF Images</th>
        <th>#Parsed Figures</th>
        <th>Rendered</th>
        <th>Overlay</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <p class="small muted">Generated by rag_parser_quality_eval.py</p>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="RAG Parser Quality Evaluator")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the source PDF")
    parser.add_argument("--jsonl", required=True, type=Path, help="Path to the parsed elements JSONL")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write the report")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional limit on pages to analyze")
    args = parser.parse_args()

    summary = build_report(args.pdf, args.jsonl, args.outdir, args.max_pages)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
