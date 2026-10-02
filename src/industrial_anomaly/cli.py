"""CLI: train, evaluate, benchmark, report, predict."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import logging
import os
import sys
from contextlib import suppress
from getpass import getpass
from pathlib import Path

from .config import Config
from .data import MVTEC_CATEGORIES, category_dirs, list_images, load_rgb
from .detector import AnomalyDetector

logger = logging.getLogger("industrial_anomaly")


# ============================================================
# CONFIGURATION
# ============================================================


def _add_config_args(p: argparse.ArgumentParser) -> None:
    d = Config()

    g = p.add_argument_group("model")

    g.add_argument(
        "--image-size",
        type=int,
        default=d.image_size,
    )

    g.add_argument(
        "--pca-components",
        type=int,
        default=d.pca_components,
    )

    g.add_argument(
        "--n-train-images",
        type=int,
        default=d.n_train_images,
        help="Training images to use (<=0 means all).",
    )

    g.add_argument(
        "--coreset-ratio",
        type=float,
        default=None,
        help="Fraction of patches to keep via greedy coreset (e.g. 0.1).",
    )

    g.add_argument(
        "--local-aggregation",
        type=int,
        default=d.local_aggregation,
    )

    g.add_argument(
        "--patch-ratio",
        type=float,
        default=d.anomaly_patch_ratio,
    )

    g.add_argument(
        "--calibration-fraction",
        type=float,
        default=d.calibration_fraction,
    )

    g.add_argument(
        "--threshold-sigma",
        type=float,
        default=d.threshold_sigma,
    )

    g.add_argument(
        "--batch-size",
        type=int,
        default=d.batch_size,
    )

    g.add_argument(
        "--seed",
        type=int,
        default=d.seed,
    )

    p.add_argument(
        "--device",
        default=None,
        help="cpu | cuda (default: auto)",
    )


def _config(a: argparse.Namespace) -> Config:
    return Config(
        image_size=a.image_size,
        pca_components=a.pca_components,
        n_train_images=(a.n_train_images if a.n_train_images > 0 else None),
        coreset_ratio=a.coreset_ratio,
        local_aggregation=a.local_aggregation,
        anomaly_patch_ratio=a.patch_ratio,
        calibration_fraction=a.calibration_fraction,
        threshold_sigma=a.threshold_sigma,
        batch_size=a.batch_size,
        seed=a.seed,
    )


# ============================================================
# TRAIN
# ============================================================


def cmd_train(a: argparse.Namespace) -> int:
    det = AnomalyDetector(_config(a), a.device)

    det.fit(category_dirs(a.data_root, a.category)["train"])

    out = Path(a.model_dir or f"models/{a.category}")

    det.save(out)

    logger.info(
        "Saved model to %s (threshold=%s)",
        out,
        det.threshold,
    )

    return 0


# ============================================================
# EVALUATE
# ============================================================


def cmd_evaluate(a: argparse.Namespace) -> int:
    from .evaluation import evaluate_category
    from .visualization import plot_evaluation

    det = AnomalyDetector.load(
        a.model_dir,
        a.device,
    )

    res = evaluate_category(
        det,
        a.data_root,
        a.category,
    )

    out = Path(a.out_dir) / a.category
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (out / "metrics.json").write_text(
        json.dumps(
            res.metrics,
            indent=2,
        )
    )

    plot_evaluation(
        res,
        out / "evaluation.png",
    )

    print(
        json.dumps(
            {k: v for k, v in res.metrics.items() if k != "mean_score_by_type"},
            indent=2,
        )
    )

    return 0


# ============================================================
# BENCHMARK
# ============================================================


def cmd_benchmark(a: argparse.Namespace) -> int:
    from .evaluation import evaluate_category
    from .visualization import plot_evaluation

    cats = a.categories or list(MVTEC_CATEGORIES)

    out = Path(a.out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = _config(a)
    rows = []

    for cat in cats:
        try:
            det = AnomalyDetector(
                cfg,
                a.device,
            )

            det.fit(
                category_dirs(
                    a.data_root,
                    cat,
                )["train"]
            )

            res = evaluate_category(
                det,
                a.data_root,
                cat,
            )

        except FileNotFoundError as e:
            logger.warning(
                "Skipping %s: %s",
                cat,
                e,
            )
            continue

        plot_evaluation(
            res,
            out / cat / "evaluation.png",
        )

        (out / cat / "metrics.json").write_text(
            json.dumps(
                res.metrics,
                indent=2,
            )
        )

        if a.save_models:
            det.save(Path(a.save_models) / cat)

        rows.append(res.metrics)

        logger.info(
            "%-12s image AUROC %.4f | pixel AUROC %s",
            cat,
            res.metrics["image_auroc"],
            res.metrics["pixel_auroc"],
        )

    if not rows:
        logger.error("No categories evaluated.")
        return 1

    fields = [
        "category",
        "image_auroc",
        "pixel_auroc",
        "image_ap",
        "f1",
        "false_alarm_rate",
        "miss_rate",
    ]

    with open(
        out / "results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        w.writeheader()
        w.writerows(rows)

    mean_img = sum(r["image_auroc"] for r in rows) / len(rows)

    pix = [r["pixel_auroc"] for r in rows if r["pixel_auroc"] is not None]

    print(f"\nMean image AUROC over {len(rows)} categories: {mean_img:.4f}")

    if pix:
        print(f"Mean pixel AUROC: {sum(pix) / len(pix):.4f}")

    return 0


# ============================================================
# PREDICT
# ============================================================


def cmd_predict(a: argparse.Namespace) -> int:
    from .inspection_report import (
        Record,
        make_thumb,
        model_fingerprint,
        sha256_file,
        write_report,
    )
    from .scoring import hotspot_summary
    from .visualization import save_side_by_side

    paths: list[Path] = []

    for item in a.images:
        p = Path(item)

        if not p.exists():
            logger.error(
                "Path does not exist: %s",
                p,
            )
            return 1

        paths += list_images(p) if p.is_dir() else [p]

    if not paths:
        logger.error("No images found in the given paths.")
        return 1

    det = AnomalyDetector.load(
        a.model_dir,
        a.device,
    )

    reporter = None

    if a.report:
        from getpass import getpass

        from .report import (
            GeminiReporter,
            ReportError,
        )

        print()
        print("=" * 60)
        print("Gemini AI Inspection Analysis")
        print("=" * 60)
        print("The API key is used only for this execution and is not saved.")
        print()

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if not api_key and sys.stdin.isatty():
            api_key = getpass("Enter Gemini API key: ").strip()

        if api_key:
            try:
                reporter = GeminiReporter(
                    api_key=api_key,
                    model=a.llm_model,
                )

                print("✓ Gemini AI enabled.")
                print()

            except ReportError as e:
                logger.warning(
                    "%s Continuing without AI notes.",
                    e,
                )

        else:
            print("No GEMINI_API_KEY configured. Continuing without Gemini analysis.")
            print()
    out = Path(a.out_dir)

    (out / "images").mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[Record] = []

    for i, pred in enumerate(det.predict_paths(paths)):
        img = load_rgb(pred.path)

        viz = det.render(
            img,
            pred,
        )

        verdict = {
            True: "FAIL",
            False: "PASS",
            None: "UNKNOWN",
        }[pred.is_anomalous]

        # MVTec reuses names like 000.png
        # in every defect folder.
        name = f"{i:04d}_{pred.path.parent.name}_{pred.path.stem}"

        save_side_by_side(
            img,
            viz,
            f"{verdict}  score={pred.score:.3f}",
            out / "images" / f"{name}.png",
        )

        hotspot = None

        if pred.is_anomalous:
            hs = hotspot_summary(det.normalized_map(pred))

            if hs:
                hotspot = (
                    f"Largest hot region: "
                    f"x={hs['x']}, "
                    f"y={hs['y']}, "
                    f"{hs['w']}x{hs['h']} px "
                    f"({hs['area_pct']}% of image)"
                )

        note = None

        if reporter and pred.is_anomalous:
            try:
                note = reporter.generate(
                    img,
                    viz,
                    pred.score,
                    det.threshold,
                    a.component,
                    True,
                )

                (out / "images" / f"{name}_note.md").write_text(
                    note,
                    encoding="utf-8",
                )

            except Exception as e:
                logger.error(
                    "AI note failed for %s: %s",
                    pred.path.name,
                    e,
                )

                note = f"AI note unavailable: {e}"

        records.append(
            Record(
                image=str(pred.path),
                sha256=sha256_file(pred.path),
                score=pred.score,
                verdict=verdict,
                thumb_b64=make_thumb(viz),
                hotspot=hotspot,
                llm_text=note,
            )
        )

    meta = {
        "component": a.component,
        "model_dir": str(a.model_dir),
        "model_id": model_fingerprint(a.model_dir),
        "threshold": (
            None
            if det.threshold is None
            else round(
                det.threshold,
                6,
            )
        ),
        "image_size": det.cfg.image_size,
        "pca_components": det.cfg.pca_components,
        "device": str(det.extractor.device),
    }

    html_path, csv_path = write_report(
        out,
        records,
        meta,
    )

    (out / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "image": r.image,
                    "score": r.score,
                    "verdict": r.verdict,
                }
                for r in records
            ],
            indent=2,
        )
    )

    for r in records:
        print(f"{r.verdict:7s} {r.score:.4f}  {r.image}")

    print(f"\nReport: {html_path}")

    print(f"Audit log: {csv_path}")

    return 0


# ============================================================
# REPORT
# ============================================================


def cmd_report(a: argparse.Namespace) -> int:
    """Generate a self-contained MVTec HTML benchmark report."""

    results_dir = Path(a.results_dir)

    csv_path = results_dir / "results.csv"

    out_path = Path(a.out)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not csv_path.exists():
        logger.error(
            "Missing benchmark results: %s",
            csv_path,
        )
        return 1

    rows: list[dict] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if not rows:
        logger.error("results.csv contains no benchmark rows.")
        return 1

    # --------------------------------------------------------
    # Metric helper
    # --------------------------------------------------------

    def metric(name: str) -> list[float]:
        values = []

        for row in rows:
            value = row.get(name)

            if value in (
                None,
                "",
                "None",
                "null",
            ):
                continue

            with suppress(ValueError):
                values.append(float(value))
        return values

    image_auroc = metric("image_auroc")

    pixel_auroc = metric("pixel_auroc")

    mean_image = sum(image_auroc) / len(image_auroc) if image_auroc else None

    mean_pixel = sum(pixel_auroc) / len(pixel_auroc) if pixel_auroc else None

    # --------------------------------------------------------
    # Gemini benchmark analysis
    # --------------------------------------------------------

    gemini_analysis = (
        "Gemini analysis was not requested." if a.no_gemini else "Gemini analysis unavailable."
    )

    if not a.no_gemini:
        print()
        print("=" * 60)
        print("Gemini AI Benchmark Analysis")
        print("=" * 60)
        print("The API key is used only for this execution and is not saved.")
        print()

        api_key = getpass("Enter Gemini API key: ").strip()

        if not api_key:
            print("No API key entered. Continuing without Gemini analysis.")

        else:
            try:
                from .report import GeminiReporter

                reporter = GeminiReporter(
                    api_key=api_key,
                    model=a.llm_model,
                )

                # Clean metric strings first.
                mean_image_text = f"{mean_image:.4f}" if mean_image is not None else "unavailable"

                mean_pixel_text = f"{mean_pixel:.4f}" if mean_pixel is not None else "unavailable"

                summary_lines = [
                    "MVTec AD benchmark results:",
                    f"Categories evaluated: {len(rows)}",
                    f"Mean image AUROC: {mean_image_text}",
                    f"Mean pixel AUROC: {mean_pixel_text}",
                    "",
                    "Per-category results:",
                ]

                for row in rows:
                    summary_lines.append(
                        f"- "
                        f"{row.get('category', 'unknown')}: "
                        f"image AUROC="
                        f"{row.get('image_auroc', 'N/A')}, "
                        f"pixel AUROC="
                        f"{row.get('pixel_auroc', 'N/A')}, "
                        f"image AP="
                        f"{row.get('image_ap', 'N/A')}, "
                        f"F1="
                        f"{row.get('f1', 'N/A')}"
                    )

                gemini_analysis = reporter.generate_benchmark_summary("\n".join(summary_lines))

                print("✓ Gemini analysis generated.")

            except Exception as e:
                logger.warning(
                    "Gemini analysis unavailable: %s",
                    e,
                )

                gemini_analysis = "Gemini analysis could not be generated for this report."

    # --------------------------------------------------------
    # HTML image helper
    # --------------------------------------------------------

    def image_data_uri(
        path: Path,
    ) -> str | None:

        if not path.exists():
            return None

        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")

            return "data:image/png;base64," + encoded

        except OSError:
            return None

    # --------------------------------------------------------
    # Category cards
    # --------------------------------------------------------

    category_cards = []

    for row in rows:
        category = row.get(
            "category",
            "unknown",
        )

        evaluation_image = results_dir / category / "evaluation.png"

        image_uri = image_data_uri(evaluation_image)

        image_html = ""

        if image_uri:
            image_html = f"""
            <img
                src="{image_uri}"
                alt="{html.escape(category)} evaluation"
                class="evaluation-image"
            />
            """

        category_cards.append(
            f"""
            <section class="category-card">

                <h3>
                    {html.escape(category.title())}
                </h3>

                <div class="metrics">

                    <div>
                        <span>Image AUROC</span>
                        <strong>
                            {
                html.escape(
                    row.get(
                        "image_auroc",
                        "N/A",
                    )
                )
            }
                        </strong>
                    </div>

                    <div>
                        <span>Pixel AUROC</span>
                        <strong>
                            {
                html.escape(
                    row.get(
                        "pixel_auroc",
                        "N/A",
                    )
                )
            }
                        </strong>
                    </div>

                    <div>
                        <span>Image AP</span>
                        <strong>
                            {
                html.escape(
                    row.get(
                        "image_ap",
                        "N/A",
                    )
                )
            }
                        </strong>
                    </div>

                    <div>
                        <span>F1</span>
                        <strong>
                            {
                html.escape(
                    row.get(
                        "f1",
                        "N/A",
                    )
                )
            }
                        </strong>
                    </div>

                </div>

                {image_html}

            </section>
            """
        )

    safe_gemini = html.escape(gemini_analysis)

    # --------------------------------------------------------
    # HTML document
    # --------------------------------------------------------

    html_document = f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Industrial Anomaly Detection — MVTec Benchmark
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: Inter, Arial, sans-serif;
    background: #f4f6f8;
    color: #18202a;
}}

.container {{
    max-width: 1400px;
    margin: auto;
    padding: 40px;
}}

.hero {{
    background: #111827;
    color: white;
    padding: 42px;
    border-radius: 18px;
    margin-bottom: 28px;
}}

.hero h1 {{
    margin: 0 0 10px;
    font-size: 34px;
}}

.hero p {{
    margin: 0;
    color: #cbd5e1;
}}

.summary {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-bottom: 32px;
}}

.metric-card {{
    background: white;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 3px 15px rgba(0,0,0,.06);
}}

.metric-card span {{
    display: block;
    color: #64748b;
    font-size: 14px;
    margin-bottom: 8px;
}}

.metric-card strong {{
    font-size: 30px;
}}

.category-card {{
    background: white;
    padding: 24px;
    margin-bottom: 24px;
    border-radius: 14px;
    box-shadow: 0 3px 15px rgba(0,0,0,.05);
}}

.category-card h3 {{
    margin-top: 0;
}}

.metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}}

.metrics div {{
    background: #f8fafc;
    padding: 14px;
    border-radius: 10px;
}}

.metrics span {{
    display: block;
    color: #64748b;
    font-size: 12px;
    margin-bottom: 5px;
}}

.evaluation-image {{
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}}

.ai {{
    background: white;
    padding: 28px;
    border-radius: 14px;
    margin-top: 30px;
}}

.ai pre {{
    white-space: pre-wrap;
    font-family: Inter, Arial, sans-serif;
    line-height: 1.65;
}}

footer {{
    color: #64748b;
    margin-top: 30px;
    font-size: 13px;
}}

@media (max-width: 800px) {{

    .container {{
        padding: 18px;
    }}

    .summary,
    .metrics {{
        grid-template-columns: 1fr;
    }}

}}

</style>

</head>

<body>

<div class="container">

<header class="hero">

    <h1>
        Industrial Anomaly Detection
    </h1>

    <p>
        MVTec AD — Full 15-category benchmark report
    </p>

</header>

<section class="summary">

    <div class="metric-card">

        <span>
            Categories evaluated
        </span>

        <strong>
            {len(rows)}
        </strong>

    </div>

    <div class="metric-card">

        <span>
            Mean image AUROC
        </span>

        <strong>
            {f"{mean_image:.4f}" if mean_image is not None else "N/A"}
        </strong>

    </div>

    <div class="metric-card">

        <span>
            Mean pixel AUROC
        </span>

        <strong>
            {f"{mean_pixel:.4f}" if mean_pixel is not None else "N/A"}
        </strong>

    </div>

</section>

<h2>
    Category Results
</h2>

{"".join(category_cards)}

<section class="ai">

    <h2>
        Gemini AI Analysis
    </h2>

    <pre>
{safe_gemini}
    </pre>

</section>

<footer>

    Generated by Industrial Anomaly Detection CLI.

    Detector metrics remain authoritative;
    Gemini provides supplementary analysis only.

</footer>

</div>

</body>

</html>
"""

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        html_document,
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("✓ HTML report generated")
    print("=" * 60)

    print(f"Report: {out_path.resolve()}")

    print(f"Categories: {len(rows)}")

    if mean_image is not None:
        print(f"Mean image AUROC: {mean_image:.4f}")
    else:
        print("Mean image AUROC: N/A")

    if mean_pixel is not None:
        print(f"Mean pixel AUROC: {mean_pixel:.4f}")
    else:
        print("Mean pixel AUROC: N/A")

    return 0


# ============================================================
# ARGUMENT PARSER
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        prog="anomaly-detect",
        description=__doc__,
    )

    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )

    sub = p.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    t = sub.add_parser(
        "train",
        help=("Fit the memory bank on normal images and calibrate."),
    )

    t.add_argument(
        "--data-root",
        required=True,
    )

    t.add_argument(
        "--category",
        required=True,
        choices=MVTEC_CATEGORIES,
    )

    t.add_argument("--model-dir")

    _add_config_args(t)

    t.set_defaults(func=cmd_train)

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    e = sub.add_parser(
        "evaluate",
        help=("Evaluate a saved model on the test split."),
    )

    e.add_argument(
        "--data-root",
        required=True,
    )

    e.add_argument(
        "--category",
        required=True,
        choices=MVTEC_CATEGORIES,
    )

    e.add_argument(
        "--model-dir",
        required=True,
    )

    e.add_argument(
        "--out-dir",
        default="outputs/eval",
    )

    e.add_argument(
        "--device",
        default=None,
    )

    e.set_defaults(func=cmd_evaluate)

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    b = sub.add_parser(
        "benchmark",
        help=("Train + evaluate several categories, write a table."),
    )

    b.add_argument(
        "--data-root",
        required=True,
    )

    b.add_argument(
        "--categories",
        nargs="+",
        choices=MVTEC_CATEGORIES,
    )

    b.add_argument(
        "--out-dir",
        default="outputs/benchmark",
    )

    b.add_argument(
        "--save-models",
        help=("Directory to also save each trained model into."),
    )

    _add_config_args(b)

    b.set_defaults(func=cmd_benchmark)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    r = sub.add_parser(
        "report",
        help=("Generate a self-contained HTML MVTec benchmark report."),
    )

    r.add_argument(
        "--results-dir",
        required=True,
        help=("Directory containing results.csv and evaluation images."),
    )

    r.add_argument(
        "--out",
        default="reports/mvtec_report.html",
        help="Output HTML report path.",
    )

    r.add_argument(
        "--llm-model",
        default="gemini-2.5-flash-lite",
        help=("Gemini model used for benchmark analysis."),
    )

    r.add_argument(
        "--no-gemini",
        action="store_true",
        help=("Generate the report without asking for a Gemini API key."),
    )

    r.set_defaults(func=cmd_report)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    r = sub.add_parser(
        "predict",
        help=("Run a saved model on images or folders."),
    )

    r.add_argument(
        "--model-dir",
        required=True,
    )

    r.add_argument(
        "--images",
        nargs="+",
        required=True,
    )

    r.add_argument(
        "--out-dir",
        default="outputs/predict",
    )

    r.add_argument(
        "--report",
        action="store_true",
        help=("LLM report for flagged images."),
    )

    r.add_argument(
        "--component",
        default="industrial component",
    )

    r.add_argument(
        "--llm-model",
        default="gemini-2.5-flash-lite",
    )

    r.add_argument(
        "--device",
        default=None,
    )

    r.set_defaults(func=cmd_predict)

    return p


# ============================================================
# MAIN
# ============================================================


def main(
    argv: list[str] | None = None,
) -> int:

    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format=("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
