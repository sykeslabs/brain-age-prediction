import numpy as np

from scale import scale_features


def test_train_median_maps_to_zero():
    # QuantileTransformer(normal) sends the training median (its 50th
    # percentile) through norm.ppf(0.5) == 0 exactly.
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

    X_train_scaled, = scale_features(X_train)

    assert X_train_scaled[2, 0] == 0.0  # median of [1..5] is 3


def test_others_use_train_statistics_not_their_own():
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # median 3
    X_val = np.array([[3.0]])  # equals train median -> should scale to 0

    _, X_val_scaled = scale_features(X_train, X_val)

    assert X_val_scaled[0, 0] == 0.0


def test_output_shape_matches_input_shape():
    X_train = np.random.rand(20, 4)
    X_val = np.random.rand(5, 4)

    X_train_scaled, X_val_scaled = scale_features(X_train, X_val)

    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape == X_val.shape


def test_robust_to_outliers_via_ranks_not_magnitude():
    # A single extreme outlier only ever occupies the top rank/quantile --
    # unlike a mean/std scaler, it can never drag the other points' scaled
    # values toward it, because the transform only depends on each point's
    # rank among the training data, never on the outlier's actual magnitude.
    X_with_outlier = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [10000.0]])

    X_scaled, = scale_features(X_with_outlier)

    normal_points = X_scaled[:5, 0]
    assert normal_points.max() - normal_points.min() > 1.0


def test_any_number_of_extra_sets_supported():
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_val = np.array([[2.0]])
    X_test = np.array([[2.0]])

    X_train_s, X_val_s, X_test_s = scale_features(X_train, X_val, X_test)

    assert X_val_s[0, 0] == X_test_s[0, 0]
