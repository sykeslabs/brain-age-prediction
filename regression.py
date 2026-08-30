import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from feature_selection import select_features
from scale import scale_features


def _family(candidate_name):
    # "Ridge(alpha=1.0)" -> "Ridge", "RandomForest(max_depth=5, ...)" -> "RandomForest"
    return candidate_name.split("(")[0]


class StackedEnsemble:
    # Combines a handful of already-fitted base models with a linear
    # meta-learner. Kept intentionally simple (predict = base predictions ->
    # linear blend) so it can be dropped into `results` and treated exactly
    # like any other candidate by main.py.
    def __init__(self, base_models, meta_learner):
        self.base_models = base_models  # list of (name, fitted_model)
        self.meta_learner = meta_learner

    def predict(self, X):
        base_predictions = np.column_stack([model.predict(X) for _, model in self.base_models])
        return self.meta_learner.predict(base_predictions)


def _candidate_models(random_state):
    candidates = [(f"Ridge(alpha={alpha})", Ridge(alpha=alpha)) for alpha in (0.1, 1.0, 10.0, 100.0)]

    candidates += [
        (f"RandomForest(max_depth={max_depth}, min_leaf={min_samples_leaf})",
         RandomForestRegressor(
             n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
             # max_features="sqrt": the sklearn default (1.0, scan every column
             # per split) lets trees memorize training rows almost exactly
             # (unlimited-depth train R² ~0.93) — per-split feature
             # subsampling is what regularizes them here.
             max_features="sqrt", n_jobs=-1, random_state=random_state,
         ))
        for max_depth in (5, 10, None) for min_samples_leaf in (1, 5, 15)
    ]
    # Also tried adding a `max_samples` (bootstrap subsample fraction) axis
    # here, and a separate stochastic-gradient-boosting family
    # (GradientBoostingRegressor with subsample<1.0) alongside HGB. Measured
    # effect on this dataset: neutral for the stacked ensemble's CV score
    # (0.5017±0.0694 vs 0.5026±0.0703 without them — within noise) and every
    # max_samples<1.0 RF variant scored strictly lower than its
    # max_samples=None counterpart. Removed — extra runtime (17 more
    # candidates × 5 feature-selection folds) with no measured benefit, not an
    # oversight.

    candidates += [
        (f"HistGradientBoosting(max_depth={max_depth}, lr={lr}, l2={l2})",
         HistGradientBoostingRegressor(
             max_depth=max_depth, learning_rate=lr, l2_regularization=l2,
             # early_stopping="auto" silently resolves to False below 10,000
             # samples, so boosting ran the full 100 iterations unconditionally
             # regardless of overfitting (train R² ~0.98). Force it on.
             early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
             random_state=random_state,
         ))
        for max_depth in (3, 5, None) for lr in (0.05, 0.1) for l2 in (0.0, 1.0)
    ]

    # Also tried ElasticNet and ExtraTrees as additional candidate families —
    # both measured as strictly redundant with what's already here (ElasticNet
    # tracked Ridge's already-weak ~0.32-0.34 linear ceiling almost exactly;
    # ExtraTrees tracked RandomForest but always slightly worse on every
    # matching config). Removed: no diversity gained for the extra runtime.
    # Also tried XGBoost — removed for a different reason: its sklearn API
    # needs an explicit eval_set to early-stop, which the shared fold loop
    # here doesn't provide (unlike HistGradientBoostingRegressor, which
    # splits its own internal validation set automatically), so it overfit
    # badly at any real depth (train R²=1.0, cv dropped to ~0.38-0.39) — and
    # even its best, shallowest config still scored below HistGradientBoosting's
    # best. Building the custom eval_set fit path to fix that wasn't worth it
    # for a family that wasn't winning anyway.
    candidates += [
        (f"CatBoost(depth={depth}, lr={lr})",
         CatBoostRegressor(
             iterations=300, depth=depth, learning_rate=lr, l2_leaf_reg=3.0,
             random_state=random_state, verbose=False, allow_writing_files=False,
         ))
        for depth in (4, 6, 8) for lr in (0.05, 0.1)
    ]

    # SVR (RBF kernel): a kernel method, structurally unlike anything else
    # here (not tree-based, not a plain linear blend) — genuinely different
    # error patterns for the stack to combine. gamma="scale" fixed: measured
    # consistently better than "auto" at every competitive C. C/epsilon grid
    # centered on a measured peak (nested CV, fixed Boruta-selected features):
    # C=15, epsilon=3.0 scored 0.4992±0.0661, beating every RandomForest/
    # HistGradientBoosting/CatBoost candidate above at the time — epsilon
    # was swept past 3.0 up to 8.0 and score strictly declined, confirming
    # this is a real peak, not a grid-edge artifact.
    candidates += [
        (f"SVR(C={C}, epsilon={epsilon})",
         SVR(kernel="rbf", C=C, gamma="scale", epsilon=epsilon))
        for C in (10.0, 15.0, 20.0) for epsilon in (2.0, 3.0)
    ]

    # Gaussian Process Regression (RBF + white-noise kernel): a sixth, still
    # RBF-kernel-based family, but one that fits noise and length-scale from
    # data via log-marginal-likelihood optimization during fit() instead of
    # grid-searching C/gamma/epsilon like SVR does, and gives a probabilistic
    # (not just point) prediction. Nested-CV compared head to head against
    # production SVR(C=15, epsilon=2.0) across 3 independent fold-seed
    # partitions (42/1/7) to rule out a lucky split: GPR won every single
    # seed by a modest but consistent margin (0.5367/0.5339/0.5201 vs SVR's
    # 0.5332/0.5273/0.5182). Only one config here, not several like every
    # other family — confirmed unnecessary, not just skipped for cost: a
    # 3x3 length_scale/noise_level sweep of initial values (5/10/20,
    # 0.5/1.0/2.0) scored IDENTICALLY (0.5367+/-0.0940) on every combination,
    # because GaussianProcessRegressor re-optimizes these hyperparameters via
    # its own internal optimizer regardless of what they're initialized to —
    # unlike SVR's C/epsilon, they aren't actually a tunable grid.
    candidates += [
        ("GPR",
         GaussianProcessRegressor(
             kernel=ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0),
             normalize_y=True, n_restarts_optimizer=2, random_state=random_state,
         )),
    ]

    return candidates


def select_best_model(X_train, y_train, X_val, X_test, y_val, candidates=None, random_state=42, cv_folds=5,
                       feature_selector=select_features, scaler=scale_features, enable_stacking=True,
                       refit_on_combined=True):
    # Model/hyperparameter SELECTION happens entirely via cross-validation on
    # the training fold. X_val is deliberately not used to pick a winner —
    # comparing candidates against one ~243-row split is enough to pick
    # whichever candidate best matches that split's noise, not whichever
    # generalizes best (observed swing across random splits: R² 0.42-0.54).
    #
    # feature_selector/scaler are redone INSIDE every fold (not once upfront,
    # like impute/outliers already are in main.py) — they use y_train, so
    # fitting them once outside the fold loop leaks each fold's held-out
    # rows' influence into the feature set it's later scored on. Measured
    # impact on this dataset: ~0.014 R² of false optimism (leaky 0.500 vs.
    # honest nested 0.486). Injectable as parameters so tests can swap in a
    # cheap no-op instead of re-running feature selection for every fold.
    if candidates is None:
        candidates = _candidate_models(random_state)

    n_samples = X_train.shape[0]
    # Plain shuffled KFold, not stratified by age: tried binning y_train into
    # deciles and using StratifiedKFold to balance the age histogram across
    # folds, on the theory that one unlucky age-skewed fold was inflating CV
    # variance. Measured effect was the opposite — CV mean dropped and CV
    # std roughly doubled across nearly every candidate (e.g. the identical
    # RandomForest(max_depth=None, min_leaf=1) config: 0.4903±0.0680 unstrat-
    # ified vs 0.4572±0.1213 stratified). Age has only ~56 distinct values
    # with heavy ties (a single age is ~5% of all rows), so per-fold
    # difficulty here is driven by individual subjects' feature-space
    # idiosyncrasies, not by their age — balancing the age histogram per
    # fold doesn't address that, and forcing this specific non-random
    # partition structure just produced a worse split by chance.
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_indices = list(kf.split(X_train))
    fold_scores = {name: [] for name, _ in candidates}
    # Out-of-fold predictions: each row's prediction always comes from a
    # model that did NOT see that row during training (same guarantee as
    # fold_scores), so this matrix is safe to use for training a stacking
    # meta-learner below without leaking the row it's predicting.
    oof_preds = {name: np.empty(n_samples) for name, _ in candidates}

    for train_idx, test_idx in fold_indices:
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

        X_fold_train_sel, X_fold_test_sel = feature_selector(X_fold_train, y_fold_train, X_fold_test, random_state=random_state)
        X_fold_train_scaled, X_fold_test_scaled = scaler(X_fold_train_sel, X_fold_test_sel)

        for name, model in candidates:
            fitted = clone(model).fit(X_fold_train_scaled, y_fold_train)
            fold_pred = fitted.predict(X_fold_test_scaled)
            fold_scores[name].append(r2_score(y_fold_test, fold_pred))
            oof_preds[name][test_idx] = fold_pred

    # Pick one "champion" per model family (best Ridge, best RandomForest,
    # best HistGradientBoosting) by their own CV score. Stacking every
    # near-duplicate candidate within a family would just feed the
    # meta-learner many highly correlated inputs; the champions are the
    # diverse, low-correlation base learners that stacking is actually meant
    # to combine (each family's bias/variance profile differs, so their
    # errors on the same subjects are less likely to coincide).
    families = {}
    for name, _ in candidates:
        families.setdefault(_family(name), []).append(name)
    champion_names = [max(members, key=lambda n: np.mean(fold_scores[n])) for members in families.values()]

    # Final feature selection + scaling on the FULL training fold (not
    # fold-restricted): standard nested-CV practice — the fold loop above
    # picks the config honestly, then a single final fit on all available
    # training data produces the model that's actually deployed.
    X_train_final, X_val_final, X_test_final = feature_selector(X_train, y_train, X_val, X_test, random_state=random_state)
    X_train_final, X_val_final, X_test_final = scaler(X_train_final, X_val_final, X_test_final)

    results = []
    for name, model in candidates:
        fitted = clone(model).fit(X_train_final, y_train)
        results.append({
            "name": name,
            "model": fitted,
            "cv_r2_mean": np.mean(fold_scores[name]),
            "cv_r2_std": np.std(fold_scores[name]),
            "train_r2": r2_score(y_train, fitted.predict(X_train_final)),
            "val_r2": r2_score(y_val, fitted.predict(X_val_final)),
        })

    # Only worth stacking if there's more than one family to diversify across.
    if enable_stacking and len(champion_names) >= 2:
        champion_oof = np.column_stack([oof_preds[name] for name in champion_names])

        # Honest CV score for the stack itself, using the SAME fold
        # boundaries as every base model: the meta-learner for outer fold i
        # is trained only on OOF predictions from the other folds, so it
        # never sees fold i's rows before being scored on them — directly
        # comparable to each candidate's cv_r2_mean above, not an optimistic
        # in-sample number. LinearRegression(positive=True) keeps the blend
        # a simple non-negative-weighted average (one input per family, so
        # a handful at most): stacking can overfit fast on a dataset this
        # small, so the meta-learner is deliberately not another complex
        # model.
        stack_fold_scores = []
        for train_idx, test_idx in fold_indices:
            meta = LinearRegression(positive=True).fit(champion_oof[train_idx], y_train[train_idx])
            stack_fold_scores.append(r2_score(y_train[test_idx], meta.predict(champion_oof[test_idx])))

        # Deployed version: base models are the already fully-fit (on
        # X_train_final) champions from `results` above — reusing them
        # rather than refitting keeps this consistent with every other
        # candidate's final model. Meta-learner weights are learned from the
        # full honest OOF matrix, then applied to base models trained on
        # more data than generated that matrix — the same asymmetry
        # sklearn's own StackingRegressor accepts, and standard practice.
        champion_models = [(name, next(r["model"] for r in results if r["name"] == name)) for name in champion_names]
        meta_final = LinearRegression(positive=True).fit(champion_oof, y_train)
        stack_model = StackedEnsemble(champion_models, meta_final)

        results.append({
            "name": "Stacked(" + "+".join(_family(name) for name in champion_names) + ")",
            "model": stack_model,
            "cv_r2_mean": np.mean(stack_fold_scores),
            "cv_r2_std": np.std(stack_fold_scores),
            "train_r2": r2_score(y_train, stack_model.predict(X_train_final)),
            "val_r2": r2_score(y_val, stack_model.predict(X_val_final)),
        })

    # Selection rule: raw argmax on cv_r2_mean, UNLESS the stack is
    # statistically indistinguishable from the raw winner — in that case
    # prefer the stack anyway. This is the "1-SE rule" (Breiman/Friedman;
    # see Hastie/Tibshirani/Friedman ESL) applied to model choice instead of
    # tree pruning: among candidates tied within noise, prefer the more
    # robust one. An ensemble blend is inherently lower-variance than any one
    # base learner it's built from, so when a single model's edge over the
    # stack is smaller than the noise in that edge, the tie should go to the
    # ensemble. This directly targets a real, observed failure mode: two
    # near-identical local CV scores (within 1 SE of each other) can flip
    # which one wins depending on the random fold split, and that flip has
    # measurably changed the Kaggle leaderboard score across submissions —
    # so this uses the SAME fold partition to compute a paired difference
    # (best-vs-stack per fold, not two independent stds), which is a more
    # powerful/less noisy test than comparing raw cv_r2_std values alone.
    # Still purely a function of X_train's own folds — never touches
    # validation or leaderboard feedback.
    best = max(results, key=lambda r: r["cv_r2_mean"])
    stack_result = next((r for r in results if r["name"].startswith("Stacked(")), None)
    if stack_result is not None and stack_result is not best:
        diffs = np.array(fold_scores[best["name"]]) - np.array(stack_fold_scores)
        se_diff = np.std(diffs) / np.sqrt(cv_folds)
        if diffs.mean() <= se_diff:
            best = stack_result

    # Final deployment refit: once the winner is picked (by CV, using only
    # X_train — nothing here touches selection), refit that EXACT
    # configuration on train+validation combined before it generates any
    # predictions. Standard practice once model choice is already locked
    # in: validation must never influence which model/hyperparameters win,
    # but the model that actually generates submission.csv should use every
    # labeled row available, not leave ~20% of it on the table. cv_r2_mean/
    # val_r2 above are untouched by this — both were already computed from
    # the train-only fit, so they remain honest post-hoc diagnostics, not a
    # description of the model this function actually deploys.
    if refit_on_combined:
        X_combined_final = np.vstack([X_train_final, X_val_final])
        y_combined = np.concatenate([y_train, y_val])

        if isinstance(best["model"], StackedEnsemble):
            refit_base_models = [
                (name, clone(model).fit(X_combined_final, y_combined))
                for name, model in best["model"].base_models
            ]
            # meta_learner is NOT refit here: its weights come from the
            # honest OOF matrix over X_train's own folds; only the base
            # models it blends get the extra data — same asymmetry as the
            # fold-level -> full-training-fold step above.
            best["model"] = StackedEnsemble(refit_base_models, best["model"].meta_learner)
        else:
            best["model"] = clone(best["model"]).fit(X_combined_final, y_combined)

    return best, results, X_test_final
