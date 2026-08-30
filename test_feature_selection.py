import numpy as np

from feature_selection import select_features


def _signal_redundant_and_noise_dataset(n_samples=200, n_noise=10, random_state=0):
    # A handful of noise columns, not just one: the relevance filter needs a
    # realistically sized pool of irrelevant columns to be a stable test
    # (mirrors the real dataset's hundreds of features, not a 2-column toy).
    rng = np.random.RandomState(random_state)
    signal = rng.normal(size=n_samples)
    duplicate = signal + rng.normal(scale=1e-6, size=n_samples)  # near-perfect duplicate
    noise = rng.normal(size=(n_samples, n_noise))  # unrelated to y
    X = np.column_stack([signal, duplicate, noise])
    y = signal * 10 + rng.normal(scale=0.1, size=n_samples)
    return X, y


def test_redundant_duplicate_column_is_pruned_before_relevance_ranking():
    X, y = _signal_redundant_and_noise_dataset()

    # k=1: only the single best-F-score column should survive. If the
    # duplicate weren't pruned first, either it or the signal could win the
    # ranking arbitrarily -- pruning must happen before top-k relevance.
    X_selected, = select_features(X, y, k=1, random_state=42)

    assert X_selected.shape[1] == 1


def test_relevant_column_survives_top_k_over_pure_noise():
    X, y = _signal_redundant_and_noise_dataset()

    X_selected, = select_features(X, y, k=1, random_state=42)

    # the one surviving column must be the true signal (or its
    # near-duplicate, whichever redundancy pruning happened to keep), not one
    # of the 10 columns with no relationship to y at all
    np.testing.assert_allclose(X_selected[:, 0], X[:, 0], atol=1e-4)


def test_k_larger_than_available_columns_keeps_everything_without_error():
    X, y = _signal_redundant_and_noise_dataset()

    # default k=100 exceeds the 11 columns left after the duplicate is
    # pruned (signal + 10 noise) -- SelectKBest must clip to what's
    # available rather than error out.
    X_selected, = select_features(X, y, random_state=42)

    assert X_selected.shape[1] == 11


def test_others_get_same_column_mask_as_train():
    X, y = _signal_redundant_and_noise_dataset()
    X_val = X[:5] * 2  # arbitrary stand-in "validation" set, same columns

    X_train_sel, X_val_sel = select_features(X, y, X_val, k=1, random_state=42)

    assert X_train_sel.shape[1] == X_val_sel.shape[1]
    # whichever original column survived selection, X_val must carry the
    # SAME column through (scaled by the same 2x used to build X_val here)
    selected_original_index = next(
        j for j in range(X.shape[1]) if np.allclose(X_train_sel[:, 0], X[:, j])
    )
    np.testing.assert_allclose(X_val_sel[:, 0], X_val[:, selected_original_index])


def test_deterministic_with_fixed_random_state():
    X, y = _signal_redundant_and_noise_dataset()

    result_1, = select_features(X, y, random_state=42)
    result_2, = select_features(X, y, random_state=42)

    np.testing.assert_array_equal(result_1, result_2)


def test_k_is_configurable_and_monotonic():
    X, y = _signal_redundant_and_noise_dataset()

    small, = select_features(X, y, random_state=42, k=1)
    large, = select_features(X, y, random_state=42, k=5)

    # a higher k can only keep as many or more features, never fewer
    assert large.shape[1] >= small.shape[1]


def test_correlated_pair_keeps_more_relevant_feature_not_earlier_column():
    rng = np.random.RandomState(0)
    n = 200
    base = rng.normal(size=n)
    weak = base + rng.normal(scale=0.3, size=n)     # column 0: weaker link to y
    strong = base + rng.normal(scale=0.01, size=n)  # column 1: stronger link to y
    y = base * 10 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([weak, strong])

    X_selected, = select_features(X, y, random_state=42)

    # weak and strong are >0.9 correlated with each other, so one must be
    # dropped by redundancy pruning. It should be `strong` that survives
    # (higher |correlation| with y), even though `weak` appears first in
    # column order — column-order tie-breaking would have kept `weak`.
    assert X_selected.shape[1] == 1
    np.testing.assert_allclose(X_selected[:, 0], strong, atol=1e-6)


def test_zero_variance_column_is_dropped_without_error():
    X, y = _signal_redundant_and_noise_dataset()
    constant_col = np.full((X.shape[0], 1), 7.0)
    X_with_constant = np.hstack([X, constant_col])

    X_selected, = select_features(X_with_constant, y, k=1, random_state=42)

    assert X_selected.shape[1] == 1  # same result as without the constant column
