import numpy as np

from scripts.build_stratified_slices import split_group, validate_ratios


def test_split_group_preserves_total_count():
    indices = np.arange(100, dtype=np.int64)
    rng = np.random.default_rng(42)
    train, validation, test = split_group(indices, rng, train_ratio=0.64, validation_ratio=0.16)
    assert len(train) + len(validation) + len(test) == 100
    assert sorted(np.concatenate([train, validation, test]).tolist()) == indices.tolist()


def test_split_group_approximate_ratios():
    indices = np.arange(10000, dtype=np.int64)
    rng = np.random.default_rng(42)
    train, validation, test = split_group(indices, rng, train_ratio=0.64, validation_ratio=0.16)
    assert abs(len(train) / 10000 - 0.64) < 0.01
    assert abs(len(validation) / 10000 - 0.16) < 0.01
    assert abs(len(test) / 10000 - 0.20) < 0.01


def test_validate_ratios_accepts_valid():
    validate_ratios(0.64, 0.16, 0.20)  # should not raise
