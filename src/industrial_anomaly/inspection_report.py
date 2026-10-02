"""Batch inspection report: one self-contained HTML file plus a CSV audit log."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import __version__


@dataclass
class Record:
    image: str
    sha256: str
    score: float
    verdict: str  # PASS | FAIL | UNKNOWN
    thumb_b64: str  # small JPEG of the heatmap overlay
    hotspot: str | None = None  # e.g. 'largest hot region at x=120,y=80, 64x40 px (0.9% of image)'
    llm_text: str | None = None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint(model_dir: str | Path) -> str:
    """Short hash of the saved model files, so a result can be traced to the exact model."""
    h = hashlib.sha256()
    for name in ("meta.json", "bank_meta.json", "pca.joblib", "index.faiss"):
        h.update(Path(model_dir, name).read_bytes())
    return h.hexdigest()[:12]


def make_thumb(rgb: np.ndarray, max_side: int = 420) -> str:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not ok:
        raise ValueError("Could not encode thumbnail")
    return base64.b64encode(buf.tobytes()).decode()


_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1100px;color:#222;padding:0 1rem}
h1{margin-bottom:.2rem} .sub{color:#666;margin-top:0}
.cards{display:flex;gap:1rem;margin:1.2rem 0;flex-wrap:wrap}
.card{border:1px solid #ddd;border-radius:8px;padding:.8rem 1.2rem;min-width:130px}
.card b{display:block;font-size:1.6rem}
table{border-collapse:collapse;width:100%} th,td{border-bottom:1px solid #eee;padding:.5rem;text-align:left;vertical-align:top}
img{border-radius:4px;max-width:220px} .PASS{color:#1a7f37;font-weight:700}
.FAIL{color:#cf222e;font-weight:700} .UNKNOWN{color:#9a6700;font-weight:700}
pre{white-space:pre-wrap;font-family:inherit;margin:.4rem 0 0}
.meta td:first-child{color:#666;width:220px} code{font-size:.85em}
"""


def write_report(out_dir: str | Path, records: list[Record], meta: dict) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    esc = html.escape

    n = len(records)
    n_fail = sum(r.verdict == "FAIL" for r in records)
    n_pass = sum(r.verdict == "PASS" for r in records)
    ordered = sorted(records, key=lambda r: r.score, reverse=True)

    rows = []
    for r in ordered:
        detail = f"<div>{esc(r.hotspot)}</div>" if r.hotspot else ""
        if r.llm_text:
            detail += f"<details><summary>AI inspection note (advisory)</summary><pre>{esc(r.llm_text)}</pre></details>"
        rows.append(
            f"<tr><td><img src='data:image/jpeg;base64,{r.thumb_b64}' alt='overlay'></td>"
            f"<td>{esc(Path(r.image).name)}<br><code>sha256 {r.sha256[:12]}</code></td>"
            f"<td>{r.score:.4f}</td><td class='{r.verdict}'>{r.verdict}</td><td>{detail}</td></tr>"
        )

    meta_rows = "".join(
        f"<tr><td>{esc(str(k))}</td><td>{esc(str(v))}</td></tr>" for k, v in meta.items()
    )
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Inspection report</title><style>{_CSS}</style></head><body>
<h1>Visual inspection report</h1><p class="sub">Generated {stamp}</p>
<div class="cards">
<div class="card"><b>{n}</b>inspected</div>
<div class="card"><b class="PASS">{n_pass}</b>pass</div>
<div class="card"><b class="FAIL">{n_fail}</b>fail</div>
<div class="card"><b>{(100 * n_fail / n if n else 0):.1f}%</b>fail rate</div></div>
<h2>Results (highest anomaly score first)</h2>
<table><tr><th>Overlay</th><th>Image</th><th>Score</th><th>Verdict</th><th>Notes</th></tr>
{"".join(rows)}</table>
<h2>Run details</h2><table class="meta">{meta_rows}
<tr><td>software</td><td>industrial-anomaly-detection {__version__}</td></tr>
<tr><td>python / platform</td><td>{esc(platform.python_version())} / {esc(platform.platform())}</td></tr></table>
<p class="sub">Verdicts come from the calibrated score threshold. AI notes are advisory only
and do not change a verdict.</p></body></html>"""

    html_path = out / "inspection_report.html"
    html_path.write_text(doc, encoding="utf-8")

    csv_path = out / "inspection_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_utc",
                "image",
                "sha256",
                "score",
                "threshold",
                "verdict",
                "model_id",
                "hotspot",
            ]
        )
        for r in records:
            w.writerow(
                [
                    stamp,
                    r.image,
                    r.sha256,
                    f"{r.score:.6f}",
                    meta.get("threshold"),
                    r.verdict,
                    meta.get("model_id"),
                    r.hotspot or "",
                ]
            )
    return html_path, csv_path
