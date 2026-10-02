import numpy as np

from industrial_anomaly.memory_bank import MemoryBank, greedy_coreset


def _features(n=400, d=32, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_training_points_have_zero_distance_and_outliers_do_not():
    f = _features()
    bank = MemoryBank(n_components=8)
    bank.fit(f)
    assert bank.search(f[:20]).max() < 1e-3
    assert bank.search(f[:20] + 25.0).min() > 1.0


def test_save_load_roundtrip(tmp_path):
    f = _features()
    bank = MemoryBank(n_components=8)
    bank.fit(f)
    bank.save(tmp_path)
    loaded = MemoryBank.load(tmp_path)
    q = _features(10, seed=1)
    np.testing.assert_allclose(bank.search(q), loaded.search(q), rtol=1e-5)


def test_coreset_size_and_determinism():
    f = _features()
    a = greedy_coreset(f, 40, seed=3)
    b = greedy_coreset(f, 40, seed=3)
    assert len(set(a.tolist())) == 40
    np.testing.assert_array_equal(a, b)


def test_coreset_bank_is_smaller_but_still_flags_outliers():
    f = _features(1000)
    bank = MemoryBank(n_components=8, coreset_ratio=0.1)
    bank.fit(f)
    assert bank.size == 100
    assert bank.search(f[:50] + 25.0).min() > bank.search(f[:50]).max()
