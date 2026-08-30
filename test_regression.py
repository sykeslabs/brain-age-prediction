import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge

from regression import select_best_model


def _linear_dataset(n_train=150, n_val=50, n_test=30, n_features=5, random_state=0):
    rng = np.random.RandomState(random_state)
    true_coefs = rng.normal(size=n_features)

    def make(n):
        X = rng.normal(size=(n, n_features))
        y = X @ true_coefs + rng.normal(scale=0.1, size=n)
        return X, y

    X_train, y_train = make(n_train)
    X_val, y_val = make(n_val)
    X_test, _ = make(n_test)
    return X_train, y_train, X_val, y_val, X_test


def _identity_selector(X_train, y_train, *others, **kwargs):
    return (X_train,) + others


def _identity_scaler(X_train, *others):
    return (X_train,) + others


# A tiny, fast candidate list: enough to exercise the selection LOGIC (pick
# by cv_r2_mean, not val_r2; report every candidate; be deterministic)
# without re-running the full 25-candidate production grid in every test.
_FAST_CANDIDATES = [
    ("DummyMean", DummyRegressor(strategy="mean")),
    ("Ridge(alpha=1.0)", Ridge(alpha=1.0)),
    ("LinearRegression", LinearRegression()),
]


@pytest.fixture(scope="module")
def selection():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()
    # identity feature_selector/scaler: this file tests the SELECTION logic
    # (cv-based picking, fold nesting, reporting), not feature-selection or
    # scaling correctness — those already have their own test files.
    best, results, X_test_final = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=_FAST_CANDIDATES,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )
    return best, results, X_val, y_val, X_test, X_test_final


def test_selects_a_well_fitting_model_on_an_easy_linear_dataset(selection):
    best, _, _, _, _, _ = selection

    assert best["cv_r2_mean"] > 0.8


def test_results_report_every_candidate_with_cv_and_val_scores(selection):
    _, results, _, _, _, _ = selection

    # +1: stacking is enabled by default and _FAST_CANDIDATES has 3 distinct
    # single-member families (DummyMean, Ridge, LinearRegression), so exactly
    # one "Stacked(...)" entry is appended on top of the 3 base candidates.
    assert len(results) == len(_FAST_CANDIDATES) + 1
    for r in results:
        assert "cv_r2_mean" in r and "cv_r2_std" in r
        assert "train_r2" in r and "val_r2" in r
        assert "model" in r and hasattr(r["model"], "predict")


def test_best_is_the_max_cv_r2_among_results_not_val_r2(selection):
    best, results, _, _, _, _ = selection

    # selection criterion is cv_r2_mean — val_r2 is diagnostic only, and the
    # winner need not also have the highest val_r2 on this particular split
    assert best["cv_r2_mean"] == max(r["cv_r2_mean"] for r in results)


def test_clear_single_model_winner_is_not_overridden_by_the_tie_rule(selection):
    best, _, _, _, _, _ = selection

    # LinearRegression fits this near-noiseless linear dataset almost
    # perfectly and clearly beats every other candidate (including the
    # stack) by far more than 1 SE — the 1-SE tie-breaking rule below must
    # not kick in here; a genuine, non-noise winner still wins outright.
    assert best["name"] == "LinearRegression"


def test_stack_is_preferred_over_a_statistically_tied_single_model():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()
    # Two identical models under different family labels: their CV scores
    # tie exactly (within floating point), so raw argmax would pick whichever
    # happens to be first — the 1-SE rule should prefer the more robust
    # stacked blend over an arbitrary pick between indistinguishable champions.
    tied_candidates = [
        ("FamilyA(alpha=1.0)", Ridge(alpha=1.0)),
        ("FamilyB(alpha=1.0)", Ridge(alpha=1.0)),
    ]

    best, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=tied_candidates,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    assert best["name"].startswith("Stacked(")


def test_the_uninformative_dummy_model_loses(selection):
    _, results, _, _, _, _ = selection

    dummy = next(r for r in results if r["name"] == "DummyMean")
    winner = max(results, key=lambda r: r["cv_r2_mean"])
    assert winner["cv_r2_mean"] > dummy["cv_r2_mean"]


def test_returned_model_can_predict_on_new_data(selection):
    best, _, X_val, y_val, _, _ = selection

    predictions = best["model"].predict(X_val)

    assert predictions.shape == y_val.shape


def test_test_set_is_transformed_and_returned(selection):
    _, _, _, _, X_test, X_test_final = selection

    # identity selector/scaler here, so shape must be unchanged
    assert X_test_final.shape == X_test.shape


def test_deterministic_with_fixed_random_state():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()

    best_1, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=_FAST_CANDIDATES, random_state=42,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )
    best_2, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=_FAST_CANDIDATES, random_state=42,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    assert best_1["name"] == best_2["name"]
    assert best_1["cv_r2_mean"] == best_2["cv_r2_mean"]


def test_default_candidates_and_transforms_are_the_full_production_pipeline():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()

    _, results, X_test_final = select_best_model(X_train, y_train, X_val, X_test, y_val)

    # 4 Ridge + 9 RandomForest + 12 HistGradientBoosting + 6 CatBoost + 6 SVR
    # + 1 GPR + 1 stacked ensemble of the best-per-family champion of each
    assert len(results) == 39
    assert X_test_final.shape[0] == X_test.shape[0]  # rows preserved even if columns are selected/dropped


def test_stacking_adds_one_ensemble_candidate_combining_family_champions(selection):
    _, results, _, _, _, _ = selection

    stacked = [r for r in results if r["name"].startswith("Stacked(")]
    assert len(stacked) == 1
    assert stacked[0]["name"] == "Stacked(DummyMean+Ridge+LinearRegression)"


def test_disabling_stacking_omits_the_ensemble_candidate():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()

    _, results, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=_FAST_CANDIDATES, enable_stacking=False,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    assert len(results) == len(_FAST_CANDIDATES)
    assert not any(r["name"].startswith("Stacked(") for r in results)


def test_stacked_model_predicts_finite_values_of_the_right_shape(selection):
    _, results, X_val, y_val, _, _ = selection

    stacked = next(r for r in results if r["name"].startswith("Stacked("))
    predictions = stacked["model"].predict(X_val)

    assert predictions.shape == y_val.shape
    assert np.all(np.isfinite(predictions))


def test_stacking_with_a_single_family_is_skipped():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()
    single_family_candidates = [("Ridge(alpha=1.0)", Ridge(alpha=1.0)), ("Ridge(alpha=10.0)", Ridge(alpha=10.0))]

    _, results, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=single_family_candidates,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    # both candidates belong to the same "Ridge" family -> only one champion,
    # nothing to diversify across, so no stacking entry is added
    assert len(results) == len(single_family_candidates)
    assert not any(r["name"].startswith("Stacked(") for r in results)


def test_final_model_is_refit_on_combined_train_and_validation_data():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()
    only_dummy = [("DummyMean", DummyRegressor(strategy="mean"))]

    best, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=only_dummy,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    combined_mean = np.concatenate([y_train, y_val]).mean()
    assert best["model"].predict(X_val[:1])[0] == pytest.approx(combined_mean)
    # sanity: train-only mean must actually differ, or this test proves nothing
    assert combined_mean != pytest.approx(y_train.mean())


def test_disabling_refit_keeps_the_model_fit_on_train_only():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()
    only_dummy = [("DummyMean", DummyRegressor(strategy="mean"))]

    best, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=only_dummy, refit_on_combined=False,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    assert best["model"].predict(X_val[:1])[0] == pytest.approx(y_train.mean())


def test_cv_and_val_scores_are_unaffected_by_the_final_refit():
    X_train, y_train, X_val, y_val, X_test = _linear_dataset()
    only_dummy = [("DummyMean", DummyRegressor(strategy="mean"))]

    with_refit, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=only_dummy, refit_on_combined=True,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )
    without_refit, _, _ = select_best_model(
        X_train, y_train, X_val, X_test, y_val, candidates=only_dummy, refit_on_combined=False,
        feature_selector=_identity_selector, scaler=_identity_scaler,
    )

    # the deployed model differs (previous two tests), but the reported
    # diagnostics must not — they describe the train-only fit either way
    assert with_refit["cv_r2_mean"] == without_refit["cv_r2_mean"]
    assert with_refit["val_r2"] == without_refit["val_r2"]
    assert with_refit["train_r2"] == without_refit["train_r2"]
