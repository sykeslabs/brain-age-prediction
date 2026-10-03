from sklearn.impute import SimpleImputer


def fill_missing_values(X_train, *others):
    # Median, not mean: outliers haven't been removed yet (that's the next
    # pipeline stage), and median is robust to them.
    # keep_empty_features: without it, a column that's all-NaN in training
    # data gets silently dropped instead of filled, shifting every column
    # after it and breaking positional alignment downstream.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_filled = imputer.fit_transform(X_train)
    # every other set (validation, test, ...) only gets transformed with
    # train-fit statistics, never refit, so none of their values leak in
    others_filled = tuple(imputer.transform(X) for X in others)
    return (X_train_filled,) + others_filled
