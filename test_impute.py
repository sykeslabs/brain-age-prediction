import numpy as np

from impute import fill_missing_values


def test_no_missing_values_unchanged():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[7.0, 8.0]])

    train_filled, test_filled = fill_missing_values(X_train, X_test)

    np.testing.assert_array_equal(train_filled, X_train)
    np.testing.assert_array_equal(test_filled, X_test)


def test_fills_train_missing_with_column_median():
    X_train = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, 10.0]])
    X_test = np.array([[np.nan, np.nan]])

    train_filled, _ = fill_missing_values(X_train, X_test)

    # column 0 median of observed values [1, 3] -> 2, column 1 has one observed value -> 10
    assert train_filled[2, 0] == 2.0
    assert train_filled[0, 1] == 10.0
    assert train_filled[1, 1] == 10.0


def test_test_set_uses_train_median_not_test_median():
    X_train = np.array([[1.0], [2.0], [3.0]])  # median 2.0
    X_test = np.array([[np.nan], [100.0], [200.0]])  # test median would be ~150

    _, test_filled = fill_missing_values(X_train, X_test)

    assert test_filled[0, 0] == 2.0


def test_no_nans_remain_in_output():
    X_train = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    X_test = np.array([[np.nan, np.nan], [5.0, 6.0]])

    train_filled, test_filled = fill_missing_values(X_train, X_test)

    assert not np.isnan(train_filled).any()
    assert not np.isnan(test_filled).any()


def test_output_shape_matches_input_shape():
    X_train = np.random.rand(10, 5)
    X_test = np.random.rand(4, 5)
    X_train[0, 0] = np.nan
    X_test[1, 2] = np.nan

    train_filled, test_filled = fill_missing_values(X_train, X_test)

    assert train_filled.shape == X_train.shape
    assert test_filled.shape == X_test.shape


def test_transforms_any_number_of_extra_sets_with_train_statistics():
    X_train = np.array([[1.0], [2.0], [3.0]])  # median 2.0
    X_val = np.array([[np.nan]])
    X_test = np.array([[np.nan]])

    train_filled, val_filled, test_filled = fill_missing_values(X_train, X_val, X_test)

    assert val_filled[0, 0] == 2.0
    assert test_filled[0, 0] == 2.0
    np.testing.assert_array_equal(train_filled, X_train)


def test_column_missing_in_every_train_row_is_not_dropped():
    # Regression test: SimpleImputer silently drops all-NaN columns unless
    # keep_empty_features=True, which would shift every later column.
    X_train = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    X_test = np.array([[4.0, np.nan]])

    train_filled, test_filled = fill_missing_values(X_train, X_test)

    assert train_filled.shape == X_train.shape
    assert test_filled.shape == X_test.shape
