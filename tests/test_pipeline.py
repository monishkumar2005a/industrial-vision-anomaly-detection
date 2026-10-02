import json

import numpy as np
import pytest

from industrial_anomaly import AnomalyDetector
from industrial_anomaly.cli import main
from industrial_anomaly.data import list_images, load_rgb
from industrial_anomaly.evaluation import evaluate_category


def test_calibration_produces_threshold_and_scale(fitted):
    assert fitted.threshold is not None and np.isfinite(fitted.threshold)
    assert fitted.map_hi > fitted.map_lo


def test_defects_score_higher_than_good(fitted, mvtec_root):
    good = [
        p.score for p in fitted.predict_paths(list_images(mvtec_root / "bottle/test/good"), False)
    ]
    bad = [
        p.score for p in fitted.predict_paths(list_images(mvtec_root / "bottle/test/broken"), False)
    ]
    assert np.mean(bad) > np.mean(good)


def test_prediction_shapes_and_overlay(fitted, mvtec_root):
    img = load_rgb(list_images(mvtec_root / "bottle/test/broken")[0])
    pred = fitted.predict_arrays([img])[0]
    assert pred.anomaly_map.shape == img.shape[:2]
    viz = fitted.render(img, pred)
    assert viz.shape == img.shape and viz.dtype == np.uint8


def test_save_load_gives_identical_scores(fitted, mvtec_root, tmp_path):
    fitted.save(tmp_path / "m")
    loaded = AnomalyDetector.load(tmp_path / "m", device="cpu")
    assert loaded.threshold == fitted.threshold
    # untrained backbone, so copy the weights over to compare fairly
    loaded.extractor.model.load_state_dict(fitted.extractor.model.state_dict())
    img = load_rgb(list_images(mvtec_root / "bottle/test/broken")[0])
    a, b = fitted.predict_arrays([img])[0], loaded.predict_arrays([img])[0]
    assert a.score == pytest.approx(b.score, rel=1e-5)


def test_evaluate_category_reports_image_and_pixel_metrics(fitted, mvtec_root):
    res = evaluate_category(fitted, mvtec_root, "bottle")
    m = res.metrics
    assert m["n_good"] == 6 and m["n_defect"] == 6
    assert 0.0 <= m["image_auroc"] <= 1.0
    assert m["pixel_auroc"] is not None and 0.0 <= m["pixel_auroc"] <= 1.0
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 12


def test_too_few_calibration_images_is_rejected(mvtec_root, tiny_config):
    from dataclasses import replace

    det = AnomalyDetector(replace(tiny_config, calibration_fraction=0.01), device="cpu")
    with pytest.raises(ValueError, match="Calibration"):
        det.fit(mvtec_root / "bottle/train/good")


def test_missing_train_dir_raises(tiny_config, tmp_path):
    with pytest.raises(FileNotFoundError):
        AnomalyDetector(tiny_config, device="cpu").fit(tmp_path / "nope")


def test_cli_train_then_evaluate(mvtec_root, tmp_path, monkeypatch):
    # the CLI asks for pretrained weights, patch that so the test stays offline
    from dataclasses import replace

    import industrial_anomaly.cli as cli

    real = cli._config
    monkeypatch.setattr(cli, "_config", lambda a: replace(real(a), pretrained=False))
    model_dir, out_dir = tmp_path / "model", tmp_path / "eval"
    common = ["--data-root", str(mvtec_root), "--category", "bottle"]
    train_args = [
        "train",
        *common,
        "--model-dir",
        str(model_dir),
        "--image-size",
        "64",
        "--pca-components",
        "16",
        "--n-train-images",
        "0",
        "--calibration-fraction",
        "0.25",
        "--device",
        "cpu",
    ]
    assert main(train_args) == 0
    assert (model_dir / "meta.json").exists()
    # evaluate loads the saved config (pretrained=False), so no download
    assert (
        main(
            [
                "evaluate",
                *common,
                "--model-dir",
                str(model_dir),
                "--out-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    metrics = json.loads((out_dir / "bottle" / "metrics.json").read_text())
    assert "image_auroc" in metrics and (out_dir / "bottle" / "evaluation.png").exists()


def test_predict_writes_report_and_unique_names(fitted, mvtec_root, tmp_path, monkeypatch):
    # same file name (000.png) lives in two folders, like real MVTec
    fitted.save(tmp_path / "model")
    out = tmp_path / "out"
    folders = [str(mvtec_root / "bottle/test/good"), str(mvtec_root / "bottle/test/broken")]
    assert main(["predict", "--model-dir", str(tmp_path / "model"), "--images", *folders,
                 "--out-dir", str(out), "--device", "cpu"]) == 0  # fmt: skip
    html = (out / "inspection_report.html").read_text(encoding="utf-8")
    assert "Visual inspection report" in html and "sha256" in html
    csv_rows = (out / "inspection_log.csv").read_text().strip().splitlines()
    assert len(csv_rows) == 13  # header + 12 images
    assert len(list((out / "images").glob("*.png"))) == 12  # none overwritten


def test_predict_missing_path_exits_cleanly(tmp_path):
    code = main(["predict", "--model-dir", str(tmp_path), "--images", str(tmp_path / "nope")])
    assert code == 1


def test_report_flag_without_api_key_still_works(fitted, mvtec_root, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fitted.save(tmp_path / "model")
    code = main(
        ["predict", "--model-dir", str(tmp_path / "model"), "--report", "--device", "cpu",
         "--images", str(mvtec_root / "bottle/test/broken"), "--out-dir", str(tmp_path / "out")]
    )  # fmt: skip
    assert code == 0
    assert (tmp_path / "out" / "inspection_report.html").exists()
    assert "hotspot" in (tmp_path / "out" / "inspection_log.csv").read_text().splitlines()[0]
