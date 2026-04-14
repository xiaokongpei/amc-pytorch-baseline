import numpy as np

from scripts.build_full_slices import split_train_validation_indices


def test_split_train_validation_indices_preserves_total_count():
    indices = np.arange(10, dtype=np.int64)
    train_indices, validation_indices = split_train_validation_indices(indices, val_ratio=0.2, seed=42)
    assert len(train_indices) == 8
    assert len(validation_indices) == 2
    assert sorted(np.concatenate([train_indices, validation_indices]).tolist()) == indices.tolist()


def test_split_train_validation_indices_zero_ratio():
    indices = np.arange(5, dtype=np.int64)
    train_indices, validation_indices = split_train_validation_indices(indices, val_ratio=0.0, seed=42)
    assert len(train_indices) == 5
    assert len(validation_indices) == 0
