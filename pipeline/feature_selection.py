import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


def _prune_correlated_features(X, y, threshold):
    # Zero-variance columns give an undefined (NaN) correlation with anything
    # and carry no information either way, so drop them up front instead of
    # letting NaNs silently skip the correlation comparison below.
    keep_mask = np.std(X, axis=0) > 0
    X_kept = X[:, keep_mask]

    corr = np.corrcoef(X_kept, rowvar=False)
    n_features = corr.shape[0]

    # Greedy pass, but in order of DESCENDING |correlation with y| rather
    # than raw column order. Column-order tie-breaking is arbitrary: if two
    # features are near-duplicates and the less-predictive one merely
    # appears earlier in the CSV, it wins and the more-predictive twin gets
    # dropped before the relevance filter below even gets a vote. This
    # correlation with y is cheap (O(p), vs. the O(p^2) pairwise matrix we
    # already compute) and directly uses the label already available here,
    # so within a cluster of correlated features we keep the one most
    # relevant to the target instead of whichever came first.
    relevance = np.array([abs(np.corrcoef(X_kept[:, j], y)[0, 1]) for j in range(n_features)])
    relevance = np.nan_to_num(relevance)
    processing_order = np.argsort(-relevance)

    kept_indices = []
    for i in processing_order:
        if not any(abs(corr[i, j]) > threshold for j in kept_indices):
            kept_indices.append(i)

    sub_mask = np.zeros(n_features, dtype=bool)
    sub_mask[kept_indices] = True
    keep_mask[keep_mask] = sub_mask
    return keep_mask


def select_features(X_train, y_train, *others, correlation_threshold=0.9, k=100, random_state=42):
    # random_state is accepted (but unused) purely so this drops into
    # select_best_model's feature_selector(..., random_state=random_state)
    # call convention without a special case -- f_regression is deterministic.
    #
    # Top-k by F-test score (k=100), not the Boruta shadow-feature test this
    # replaced: once SVR became the strongest model family (see regression.py),
    # a nested-CV comparison against SVR specifically -- re-verified across 3
    # independent CV fold partitions to rule out a lucky split -- found this
    # univariate filter beating Boruta by a large, consistent margin on every
    # seed (cv 0.5098/0.4875/0.4896 for Boruta vs 0.5290/0.5285/0.5147 for
    # k=100 f_regression, a +0.02 to +0.04 R² swing each time). Boruta's
    # RandomForest-importance criterion favors whatever a tree ensemble finds
    # useful (including features that only pay off through splits/interactions);
    # SVR's RBF kernel instead rewards features with a clean univariate
    # relationship to the target, which is exactly what f_regression measures
    # directly. k=100 was the consistent best (or tied-best) of {75, 100, 125,
    # 150} across all 3 seeds tested.
    redundancy_mask = _prune_correlated_features(X_train, y_train, correlation_threshold)
    X_train_pruned = X_train[:, redundancy_mask]

    k_eff = min(k, X_train_pruned.shape[1])
    relevance_mask = SelectKBest(f_regression, k=k_eff).fit(X_train_pruned, y_train).get_support()

    final_mask = redundancy_mask.copy()
    final_mask[redundancy_mask] = relevance_mask

    others_selected = tuple(X[:, final_mask] for X in others)
    return (X_train[:, final_mask],) + others_selected
