import numpy as np

from industrial_anomaly.scoring import anomaly_map, image_score, normalize_map


def test_image_score_is_mean_of_top_k():
    d = np.arange(100, dtype=np.float32)
    assert image_score(d, 0.05) == np.mean([95, 96, 97, 98, 99])


def test_image_score_uses_at_least_one_patch():
    d = np.array([1.0, 2.0, 9.0], dtype=np.float32)
    assert image_score(d, 0.001) == 9.0  # the old int() version averaged every patch here


def test_anomaly_map_shape_and_localisation():
    d = np.zeros(64, dtype=np.float32)
    d[0] = 10.0  # top-left patch of an 8x8 grid
    m = anomaly_map(d, (8, 8), (80, 120), sigma=2.0)
    assert m.shape == (80, 120)
    assert np.unravel_index(m.argmax(), m.shape)[0] < 20  # hot spot stays in the top rows


def test_normalize_map_is_clipped_and_fixed_scale():
    m = np.array([[0.0, 5.0], [10.0, 20.0]], dtype=np.float32)
    n = normalize_map(m, lo=0.0, hi=10.0)
    assert n.min() == 0.0 and n.max() == 1.0
    # A uniformly "normal" image must stay cold, not be stretched to [0, 1] per image.
    assert normalize_map(np.full((4, 4), 1.0, np.float32), 0.0, 10.0).max() < 0.2


def test_hotspot_summary_finds_the_blob():
    from industrial_anomaly.scoring import hotspot_summary

    m = np.zeros((100, 200), np.float32)
    m[20:40, 50:90] = 1.0  # 20 rows x 40 cols
    hs = hotspot_summary(m)
    assert (hs["x"], hs["y"], hs["w"], hs["h"]) == (50, 20, 40, 20)
    assert hotspot_summary(np.zeros((10, 10), np.float32)) is None
