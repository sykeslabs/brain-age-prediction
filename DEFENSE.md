# 15-Minute Defense Walkthrough — Brain Age Prediction

A timed script for presenting and defending this project. Each section has a target duration, the
talking points to hit, the exact numbers to have ready, and likely questions with prepared answers.
Total: 15 minutes (12 min walkthrough + 3 min buffer/Q&A lead-in). Full Q&A is assumed to follow
separately — Section 8 preps you for it.

Companion document: `DOCUMENTATION.md` has the full, plain-language explanation of every
experiment referenced here. This file is the *presentation script*; that file is the *reference*.

---

## Timing at a glance

| # | Section | Time | Cumulative |
|---|---|---|---|
| 1 | Problem & why it's hard | 1.5 min | 1.5 min |
| 2 | Pipeline overview (the 6 stages) | 1 min | 2.5 min |
| 3 | Outlier handling (cell- and row-level) | 2 min | 4.5 min |
| 4 | Imputation | 1 min | 5.5 min |
| 5 | Feature selection | 2.5 min | 8 min |
| 6 | Scaling | 1 min | 9 min |
| 7 | Model selection, stacking, final refit | 3 min | 12 min |
| 8 | Results, honesty of the evaluation, what we'd try next | 2 min | 14 min |
| — | Buffer / transition to open Q&A | 1 min | 15 min |

---

## 1. Problem & why it's hard (1.5 min)

**Say this:**
"We're predicting a person's age from 832 numeric features per subject. There's no single
feature that gives it away — the best single feature only reaches a correlation of 0.44 with age,
and none exceed 0.5. Only about 6% of features even exceed 0.3. The features also barely relate to
*each other* — average pairwise correlation is 0.049 — and the top 50 principal components only
capture 32% of total variance. So this is a genuinely diffuse-signal problem: no hidden strong
predictor we're missing, just many weak, largely independent clues we need to combine carefully.
The target itself, age, is close to symmetric (skew ≈ -0.2), so we didn't need any target
transform like a log."

**Numbers to have ready:** max |corr| = 0.44; 49/832 features (5.9%) above |corr| > 0.3; avg
pairwise feature correlation = 0.049; top-50 PCA variance = 32%; y skew ≈ -0.2, 56 unique values.

**Why it matters for the rest of the talk:** this framing justifies almost every later decision —
why we didn't chase a target transform, why tree-based "interaction-hunting" feature selection
(Boruta) ended up losing to a simple univariate filter, and why PCA wasn't pursued as a feature
representation.

---

## 2. Pipeline overview (1 min)

**Say this, while pointing at a diagram or the stage list:**
"The pipeline runs as six sequential stages, each isolated in its own module: split off a
validation set → fix single corrupted cell values → fill in missing values → remove anomalous
whole rows → select and de-duplicate features → scale → try 6 model families and pick the best,
honestly, via cross-validation."

**One sentence per stage, in order** (use this as your speaking outline, don't read all of it —
each gets its own deep-dive section below):

1. `split.py` — hold out 10% as validation, before anything else touches the data.
2. `outliers.py: remove_cell_outliers` — fix individual broken cell values (IQR bounds, k=2.5).
3. `impute.py` — fill NaNs with per-column median, fit on train only.
4. `outliers.py: remove_outliers` — drop whole anomalous training rows (IsolationForest).
5. `feature_selection.py` — de-duplicate correlated features, then keep top 100 by F-test.
6. `scale.py` + `regression.py` — QuantileTransformer, then 6 model families + stacking.

---

## 3. Outlier handling: cell-level vs. row-level (2 min)

**Say this:**
"We handle two *different* kinds of corruption, and this distinction is the key design point here.
Cell-level: a handful of individual (row, feature) values were replaced with implausible numbers,
scattered across many different rows. Row-level: a whole subject's overall feature pattern looks
anomalous.

We correct cell-level outliers *first*, using per-column IQR bounds with `k=2.5`, fit on the
training fold only but applied to train, validation, *and* test alike — because correcting one bad
cell in an otherwise-fine row is not the same as discarding that row's prediction. Flagged cells
become NaN and get filled by the same imputer as any naturally-missing value.

We only added this step because IsolationForest — our row-level detector — is nearly blind to
single-cell corruption: one broken value among 832 columns barely shifts a row's overall anomaly
score. That's confirmed by IsolationForest only flagging about 4 of 969 rows — cell-level
corruption needed its own, separate mechanism.

Row-level outlier removal happens later, after imputation, and only ever touches the *training*
set — validation and test rows always get scored, since in the real evaluation every row needs a
prediction."

**Numbers to have ready:**
- Cell-level: `k=2.5` chosen over the more standard `k=3.0` — tighter std (0.0463 vs. 0.0688) and
  higher mean R², swept 2.0–4.0.
- Row-level: `contamination="auto"` removes ~4/969 rows; `0.02` was statistically tied (0.492 vs.
  0.492) but higher values (0.05, 0.1) were consistently worse — auto was already near-optimal.

**Anticipate:** "Why not use one method for both?" → Answer: a single corrupted cell doesn't move a
whole-row multivariate anomaly score enough to trigger row-level detection; they're solving
different problems and need different granularities.

---

## 4. Imputation (1 min)

**Say this:**
"Missing values are filled using per-column median, fit on the training fold only and reused
(never refit) on validation and test. Median instead of mean because outlier removal hasn't
happened yet at this point — median resists being dragged by extremes.

We tried a smarter, cross-feature-aware alternative, `KNNImputer` — filling a blank by looking at
similar *rows*. It measured worse: 0.4537 ± 0.0210 vs. plain median's 0.4656 ± 0.0688 nested CV R².
This is consistent with Section 1's finding that features are nearly mutually uninformative here —
there isn't much real row-to-row similarity for a KNN approach to exploit, so it just adds noise."

**Numbers to have ready:** median 0.4656 ± 0.0688 vs. KNN 0.4537 ± 0.0210 (nested CV R²).

**Anticipate:** "Isn't KNN supposed to be strictly better?" → No — it depends on features actually
carrying cross-feature signal, which Section 1's correlation numbers show they mostly don't here.

---

## 5. Feature selection (2.5 min — the most defensible section, spend real time here)

**Say this:**
"This is a two-step process. First, we prune near-duplicate features: any pair correlating above
0.9 gets collapsed to one. We process features in descending order of their own correlation with
age — not raw column order — so that within a duplicate cluster, we keep whichever twin is more
predictive, not whichever happened to be typed into the CSV first. Second, of what's left, we keep
the top 100 by a univariate F-test (`f_regression`).

This F-test approach replaced an earlier Boruta-style method — compare each feature's Random
Forest importance against a shuffled fake copy of itself, keep only features that reliably beat
their shadow. Boruta looked good in isolation (unanimous voting beat majority voting beat no
selection: ~0.49 vs. ~0.48 vs. ~0.41). But once SVR became our strongest model family, we re-ran
Boruta vs. the F-test filter head-to-head against SVR specifically, across 3 independent
cross-validation seeds to rule out a lucky split. The F-test filter won every single time by a
large, consistent margin: 0.5098/0.4875/0.4896 for Boruta vs. 0.5290/0.5285/0.5147 for the F-test,
a 0.02–0.04 R² swing, always in the F-test's favor.

The likely reason: Boruta's criterion rewards whatever a tree ensemble finds useful, including
features that only pay off through interactions. SVR's RBF kernel instead rewards features with a
clean, direct, univariate relationship to the target — exactly what `f_regression` measures. Once
model choice changed, the right feature selector changed with it.

We also tried mutual information (worse, ~0.45–0.49 — likely too high-variance an estimator for
a few-hundred-row fold) and Lasso-coefficient selection (worse, ~0.43–0.46, peaking at ~16
features) — both lost to the direct F-test filter."

**Numbers to have ready:**
- Correlation threshold for de-duplication: 0.9.
- Boruta vs. F-test, 3 seeds (42/1/7): Boruta 0.5098/0.4875/0.4896 vs. F-test 0.5290/0.5285/0.5147.
- k swept {75, 100, 125, 150} → 100 consistently best/tied-best.
- Mutual info ~0.45–0.49 (worse); Lasso ~0.43–0.46 (worse).

**Anticipate:** "How do you know the F-test win isn't just an artifact of SVR being the model?" →
That's explicitly why it's framed as "once SVR became the strongest family" — the feature selector
and the model family are coupled decisions here, and we re-validated the selector choice
specifically against the model that ended up winning, across 3 seeds, not just once.

---

## 6. Scaling (1 min)

**Say this:**
"Selected features get rescaled with `QuantileTransformer(output_distribution='normal')` — a
rank-based transform, not `RobustScaler`'s median/IQR approach. This only matters for SVR and
Ridge; tree-based candidates are invariant to any monotonic per-feature rescaling, so this choice
never touches their scores.

Head-to-head against SVR specifically: QuantileTransformer scored 0.5104 ± 0.0664 vs. RobustScaler's
0.4992 ± 0.0661 at the same `SVR(C=15, epsilon=3.0)` config — and re-tuning C/epsilon from scratch
under QuantileTransformer landed on that same config as its own peak, confirming this isn't an
artifact of hyperparameters chosen for the other scaler."

**Numbers to have ready:** QuantileTransformer 0.5104 ± 0.0664 vs. RobustScaler 0.4992 ± 0.0661.

---

## 7. Model selection, stacking, and final refit (3 min — second-most important section)

**Say this:**
"We evaluate 6 model families — Ridge, RandomForest, HistGradientBoosting, CatBoost, SVR, and
Gaussian Process Regression — about 38 candidates total across their hyperparameter grids.
Selection is by mean cross-validation R² on the training fold, using 5-fold KFold, shuffled,
never stratified by age — we tested stratifying by age decile and it measurably hurt: mean dropped
and standard deviation roughly doubled, because per-fold difficulty here comes from individual
subjects' feature idiosyncrasies, not from their age.

Critically, feature selection and scaling are redone **inside every single fold**, not fit once
upfront — both use the age labels, so fitting them once outside the fold loop would leak each
fold's held-out rows' influence into the very features it's later scored on. We measured this
leak directly: about 0.014 R² of false optimism, 0.500 leaky vs. 0.486 honest, same settings
otherwise.

We pick one champion per family — best Ridge, best RandomForest, and so on — and blend them with a
simple non-negative-weight linear meta-learner (`LinearRegression(positive=True)`). The stack's own
score is computed on the exact same fold boundaries as every base model, so it's directly
comparable, not an optimistic in-sample number.

Selection rule: argmax on CV mean, *unless* the stack and the raw winner are statistically tied —
specifically, if the mean paired per-fold difference between them is smaller than one standard
error of that difference, we deploy the stack anyway. This is the 1-SE rule from CART
pruning / Lasso lambda selection, applied to model choice: among options tied within noise, prefer
the lower-variance one.

This rule isn't just theoretical — it fixed a real, observed failure. Two Kaggle submissions
differing only in `cv_folds` (5 vs. 10) had their raw-argmax winner flip between the stack and a
plain RandomForest that beat it locally by just 0.0006–0.0015. The leaderboard score dropped from
0.6344 to 0.6174 on the submission where the single model won that coin-flip instead of the stack.

Finally: once the winning configuration is locked in using training-fold CV only, we refit that
exact configuration on training + validation combined before it produces any prediction — the
model that generates `submission.csv` uses every labeled row available. Validation never
influences *which* model wins; it just gets folded into the final fit once the choice is already
made."

**Numbers to have ready:**
- ~38 candidates, 6 families, `cv_folds=5`.
- Leakage measurement: leaky 0.500 vs. honest nested 0.486.
- GPR vs. SVR, 3 seeds: GPR 0.5367/0.5339/0.5201 vs. SVR 0.5332/0.5273/0.5182 (GPR wins every seed).
- SVR peak: `C=15, epsilon=3.0` at 0.4992 ± 0.0661 (measured before GPR/QuantileTransformer
  refinements — treat as a historical waypoint, not the final number).
- The 1-SE rule's real-world payoff: 0.6344 → 0.6174 leaderboard drop when it wasn't applied.
- `cv_folds` 10 vs. 5: statistically indistinguishable (differences inside ~0.09–0.10 per-fold std).

**Anticipate:** "Why not just always pick the single best model — isn't the 1-SE rule overfitting
to one anecdote?" → It's a well-established variance-reduction principle (cite CART pruning / Lasso
lambda selection), and the one anecdote we have directly demonstrates the failure mode it's
designed to prevent — a small, noise-level local edge flipping the leaderboard outcome. It's also
still purely a function of training-fold CV; it never looks at validation or leaderboard feedback,
so it doesn't compromise the honesty of the selection process, only its stability.

**Anticipate:** "Why remove XGBoost / ElasticNet / ExtraTrees?" → XGBoost needed an explicit
`eval_set` for early stopping that the shared fold loop doesn't provide, so it overfit badly
(train R²=1.0, CV ~0.38–0.39) even at its best config. ElasticNet tracked Ridge's ~0.32–0.34 linear
ceiling almost exactly; ExtraTrees always scored slightly below RandomForest on matching configs —
both were redundant, not diverse, additions.

---

## 8. Results, honesty of the evaluation, and what we'd try next (2 min)

**Say this:**
"Three numbers get reported for the selected model: cross-validation R² on the training fold (the
number actually used for selection), validation R² (a single, honest, ~243-row diagnostic — never
used to pick the winner), and the eventual Kaggle leaderboard score (the only real test score,
since `X_test.csv` has no labels locally). Target was validation R² ≥ 0.5.

We deliberately kept these three signals separate throughout: `X_val` never influences model or
hyperparameter choice — with only ~243 rows, picking a winner against one fixed split just picks
whichever candidate fits *that split's* noise best, and we saw this concretely: the same model's
score swung between R² 0.42 and 0.54 just from which rows landed in a random validation split.

If we had more time, the next things worth trying: re-sweep the Boruta `confirmation_threshold`
now that the F-test filter has replaced it as the default (we did a partial sweep and found no
reliable win, but it was against the older single-family setup); revisit `max_samples`/`subsample`
sub-1.0 axes now that the family list includes SVR and GPR, since the earlier neutral result
predates those; and settle the `cv_folds=5` vs. `10` question with a larger, dedicated sweep rather
than treating it as still-open."

**Numbers to have ready:** target validation R² ≥ 0.5; the actual `main.py` output for CV mean±std,
train R², val R², and the selected model name (pull this fresh before presenting — it changes
whenever the pipeline or data changes).

---

## Before you present: checklist

- [ ] Run `python main.py` once beforehand and note the actual printed CV/val R² and selected
      model name — don't present stale numbers from a previous run.
- [ ] Run `python -m pytest` and confirm it's green — if asked "did you test this," the answer
      should be yes, with a passing suite.
- [ ] Have `DOCUMENTATION.md` open in a second window/tab in case a question needs a deeper answer
      than this script covers.
- [ ] Know where each number above comes from in the code (`regression.py`, `feature_selection.py`,
      `outliers.py`, `scale.py`, `impute.py`, `split.py`) in case asked to point at the line.

## Likely defense questions not already covered above

- **"Why is 100 validation rows / 243 rows enough to trust as a diagnostic at all?"** — It isn't
  meant to be trusted alone; it's a sanity check, not the selection signal. The actual selection
  signal is 5-fold CV on ~870 training rows, which is why CV mean, not val_r2, drives every choice.
- **"Why age deciles for stratification and not something else?"** — Deciles were the natural
  choice for a continuous target with 56 unique values; we didn't sweep bin count separately since
  the whole approach (stratifying by age) was already shown to hurt, regardless of bin granularity.
- **"What's your actual final validation R²?"** — Pull this from the latest `python main.py` run;
  do not quote a number from this document, since it will drift as the pipeline changes.
- **"Could you have leaked test data anywhere?"** — No: `X_test.csv` never contributes to fitting
  any statistic (imputer, scaler, feature selector, or model) — it's only ever `.transform()`'d or
  `.predict()`'d, and this is enforced by explicit unit tests per module (`test_*.py` for each
  pipeline stage).
