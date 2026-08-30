import numpy as np

from split import split_train_validation


def test_split_sizes_match_val_size():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, X_val, y_train, y_val = split_train_validation(X, y, val_size=0.2, random_state=42)

    assert X_train.shape[0] == 40
    assert X_val.shape[0] == 10
    assert y_train.shape[0] == 40
    assert y_val.shape[0] == 10


def test_x_and_y_rows_stay_aligned():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, X_val, y_train, y_val = split_train_validation(X, y, val_size=0.2, random_state=42)

    for row, label in zip(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])):
        assert row[0] == label * 2  # X rows were built as [2*y, 2*y+1]


def test_same_seed_is_deterministic():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    result_1 = split_train_validation(X, y, val_size=0.2, random_state=42)
    result_2 = split_train_validation(X, y, val_size=0.2, random_state=42)

    for a, b in zip(result_1, result_2):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_splits():
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)

    _, _, _, y_val_a = split_train_validation(X, y, val_size=0.2, random_state=42)
    _, _, _, y_val_b = split_train_validation(X, y, val_size=0.2, random_state=0)

    assert not np.array_equal(sorted(y_val_a), sorted(y_val_b))


def test_train_and_validation_are_disjoint():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    _, _, y_train, y_val = split_train_validation(X, y, val_size=0.2, random_state=42)

    assert set(y_train).isdisjoint(set(y_val))
    assert set(y_train) | set(y_val) == set(y)
