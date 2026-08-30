import numpy as np

from outliers import remove_cell_outliers, remove_outliers


def _cluster_with_outliers(n_outliers):
    rng = np.random.RandomState(0)
    cluster = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
    outliers = np.full((n_outliers, 5), 1000.0)
    X = np.vstack([cluster, outliers])
    y = np.arange(X.shape[0])
    return X, y


def test_removes_known_outlier_rows():
    X, y = _cluster_with_outliers(n_outliers=2)
    contamination = 2 / X.shape[0]

    X_clean, y_clean = remove_outliers(X, y, contamination=contamination, random_state=0)

    # the two injected outliers are the last two ids (60, 61)
    assert 60 not in y_clean
    assert 61 not in y_clean


def test_x_and_y_stay_aligned():
    X, y = _cluster_with_outliers(n_outliers=2)
    contamination = 2 / X.shape[0]

    X_clean, y_clean = remove_outliers(X, y, contamination=contamination, random_state=0)

    assert X_clean.shape[0] == y_clean.shape[0]
    for row, label in zip(X_clean, y_clean):
        np.testing.assert_array_equal(row, X[label])


def test_only_removes_rows_never_adds_or_reorders_survivors():
    X, y = _cluster_with_outliers(n_outliers=2)
    contamination = 2 / X.shape[0]

    X_clean, y_clean = remove_outliers(X, y, contamination=contamination, random_state=0)

    assert X_clean.shape[0] <= X.shape[0]
    assert list(y_clean) == sorted(y_clean)  # original row order preserved
    assert set(y_clean).issubset(set(y))


def test_deterministic_with_fixed_random_state():
    X, y = _cluster_with_outliers(n_outliers=2)
    contamination = 2 / X.shape[0]

    _, y_clean_1 = remove_outliers(X, y, contamination=contamination, random_state=0)
    _, y_clean_2 = remove_outliers(X, y, contamination=contamination, random_state=0)

    np.testing.assert_array_equal(y_clean_1, y_clean_2)


def test_no_outliers_removed_when_data_is_uniform():
    X = np.ones((30, 4))
    y = np.arange(30)

    X_clean, y_clean = remove_outliers(X, y, contamination=0.01, random_state=0)

    assert X_clean.shape[0] == 30
    assert y_clean.shape[0] == 30


def _normal_column(n=100, random_state=0):
    return np.random.RandomState(random_state).normal(loc=50.0, scale=5.0, size=n)


def test_extreme_cell_value_is_flagged_as_nan():
    X = _normal_column().reshape(-1, 1)
    X[0, 0] = 10_000.0  # way outside the column's IQR-based bounds

    X_flagged, = remove_cell_outliers(X)

    assert np.isnan(X_flagged[0, 0])
    assert not np.isnan(X_flagged[1:, 0]).any()  # nothing else touched


def test_normal_values_are_not_flagged():
    X = _normal_column().reshape(-1, 1)

    X_flagged, = remove_cell_outliers(X)

    assert not np.isnan(X_flagged).any()


def test_others_are_flagged_using_train_bounds_not_their_own():
    X_train = _normal_column().reshape(-1, 1)
    # a value that is NOT extreme relative to X_train's own distribution,
    # but would look extreme if bounds were (wrongly) computed from X_val
    # itself instead of from X_train
    X_val = np.array([[10_000.0], [50.0]])

    X_train_flagged, X_val_flagged = remove_cell_outliers(X_train, X_val)

    assert np.isnan(X_val_flagged[0, 0])  # still flagged relative to train's bounds
    assert not np.isnan(X_val_flagged[1, 0])
    assert not np.isnan(X_train_flagged).any()  # train itself is untouched


def test_zero_spread_column_is_left_untouched():
    # a constant column has IQR = 0; without the zero-spread guard, every
    # value that isn't exactly Q1 would be (wrongly) flagged
    X = np.full((30, 1), 7.0)
    X[0, 0] = 7.5  # a tiny, clearly non-outlier deviation

    X_flagged, = remove_cell_outliers(X)

    assert not np.isnan(X_flagged).any()


def test_existing_nan_values_are_preserved_not_double_flagged():
    X = _normal_column().reshape(-1, 1)
    X[0, 0] = np.nan

    X_flagged, = remove_cell_outliers(X)

    assert np.isnan(X_flagged[0, 0])
    assert not np.isnan(X_flagged[1:, 0]).any()


def test_shape_is_preserved():
    X_train = _normal_column(n=50).reshape(-1, 1)
    X_val = _normal_column(n=20, random_state=1).reshape(-1, 1)

    X_train_flagged, X_val_flagged = remove_cell_outliers(X_train, X_val)

    assert X_train_flagged.shape == X_train.shape
    assert X_val_flagged.shape == X_val.shape
