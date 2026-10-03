import numpy as np
from sklearn.ensemble import IsolationForest


def remove_outliers(X_train, y_train, contamination="auto", random_state=0):
    # Row-level, multivariate: no covariance matrix (unreliable with
    # p close to n) and no distributional assumption, unlike z-score/
    # Mahalanobis-style methods.
    detector = IsolationForest(contamination=contamination, random_state=random_state)
    labels = detector.fit_predict(X_train)  # 1 = inlier, -1 = outlier
    inlier_mask = labels == 1
    return X_train[inlier_mask], y_train[inlier_mask]


def remove_cell_outliers(X_train, *others, k=2.5):
    # Complements remove_outliers, which only catches a whole ROW whose
    # entire feature vector looks anomalous. If the injected outliers are
    # instead CELL-level — a handful of individual (row, feature) values
    # replaced with extreme numbers, scattered across many different rows,
    # the same way missing values are cell-level rather than row-level — a
    # single corrupted value among hundreds of features barely moves that
    # row's overall multivariate anomaly score, so IsolationForest is nearly
    # blind to it (consistent with removing only ~4 of 969 rows).
    #
    # k=2.5, not the more conventional 3.0: sensitivity-checked via nested
    # CV across 2.0-4.0 with a fixed model — 2.5 gave both a higher mean R²
    # and a notably tighter std (0.0463 vs 0.0688 at k=3.0), i.e. tighter
    # bounds catch more real corruption without over-flagging legitimate
    # values. Per-column IQR bounds, fit on X_train only (nan-aware: this
    # runs BEFORE imputation, so the original missing values are still NaN
    # here),
    # applied to X_train and every array in `others` — unlike remove_outliers,
    # this DOES touch validation/test, since correcting one corrupted cell in
    # an otherwise-fine row is not the same as discarding that row's
    # prediction. Flagged cells become NaN, to be filled by the same median
    # imputation as any other missing value.
    q1 = np.nanpercentile(X_train, 25, axis=0)
    q3 = np.nanpercentile(X_train, 75, axis=0)
    iqr = q3 - q1

    # A zero IQR (e.g. a column dominated by one repeated value) gives no
    # reliable spread to test against — flagging everything that merely
    # differs from Q1 would be wrong, so such columns are left untouched.
    zero_spread = iqr == 0
    lower = np.where(zero_spread, -np.inf, q1 - k * iqr)
    upper = np.where(zero_spread, np.inf, q3 + k * iqr)

    def _flag(X):
        X = X.copy()
        X[(X < lower) | (X > upper)] = np.nan
        return X

    return tuple(_flag(X) for X in (X_train,) + others)
