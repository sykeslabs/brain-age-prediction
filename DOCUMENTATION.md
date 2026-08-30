# How We Made the Age-Guessing Machine Smarter

This document explains, in very simple language, every experiment we ran to make our
age-prediction pipeline better. Each section has:

- **In kid terms:** a simple explanation of the idea, no jargon.
- **The real experiment:** the actual parameters we tried, the numbers we measured, and what we
  finally picked. This part is for whoever needs the precise details (a report, a defense, a
  re-run of the experiment).

The whole point of the project: we get 832 numbers (features) about a person's brain, and we have
to guess their age. We are not told the "trick" — we have to find patterns ourselves, and the
patterns are weak and spread across many features rather than hiding in one obvious column.

---

## 0. Why this is a hard puzzle (before touching any code)

**In kid terms:** Imagine 832 different clues about a person, but every clue only whispers a
tiny, tiny hint about their age — none of them shout it. There's no single magic clue that gives
the answer away.

**The real experiment:** We measured how strongly each of the 832 features correlates with age
(`y`) directly on `X_train.csv`/`y_train.csv`:

- The single best feature only reaches `|correlation| = 0.44` with age — no feature exceeds 0.5.
- Only 49 out of 832 features (5.9%) even exceed `|correlation| > 0.3`.
- Features barely relate to *each other* either (average pairwise correlation is just 0.049), and
  squeezing everything into the top 50 PCA components only captures 32% of the total variance.
- The age values themselves (`y`) are close to a symmetric bell shape (skew ≈ -0.2, 56 distinct
  ages), so we didn't bother with tricks like log-transforming the target — there's no lopsided
  shape to fix.

This told us upfront: this is a genuinely hard, diffuse-signal problem. There's no hidden single
predictor we're missing — we need to combine many weak, independent clues carefully.

---

## 1. Splitting off a "practice test" before doing anything else

**In kid terms:** Before you start studying, you set aside a few practice questions that you
promise not to peek at while studying. That way, when you check them at the very end, you get an
honest idea of how ready you really are — not a fake grade from questions you already memorized.

**The real experiment:** We split the training data into a training set and a validation
("practice test") set **before any other step touches the data**, using `val_size=0.1` (10% held
out) and `random_state=42` for reproducibility.

Why 10% and not the more common 20%? We tested both:

- A smaller validation slice leaves more rows for the model-selection cross-validation step later
  (see Section 6), which measurably steadied that selection process — the noise (standard
  deviation) in the selection score dropped about 46% (from 0.0688 down to 0.0372), and the
  average score also rose (0.4656 → 0.4896), for one fixed test configuration.
- This costs nothing for the model we actually ship, because the very last thing we do (Section 6,
  "final refit") retrains the chosen model on training + validation data combined anyway — the
  validation rows aren't wasted, they're just not used to *pick* the model.
- Caveat, in the spirit of full honesty: this gain didn't add up perfectly once combined with
  every other change in the pipeline — in the full pipeline, that same RandomForest configuration's
  noise level actually went up slightly (0.0750 vs. 0.0372 measured in isolation). So we treat this
  as "directionally good," not as a guaranteed, stacking bonus.

The validation set is **never** used to pick which model or settings win — only to sanity-check
the winner afterward, and to help fit the final model once the choice is locked in.

---

## 2. Fixing single "broken" numbers before filling in blanks

**In kid terms:** Imagine a form where 831 boxes have normal answers, but one box says someone's
"favorite color" is one million. That's clearly a typo in one box, not a sign the whole form is
garbage. We want to spot and fix those one-off broken boxes before we do anything else with the
form.

**The real experiment:** This step (`remove_cell_outliers`) looks at each feature column on its
own and computes the normal healthy range for that column using the interquartile range (IQR) —
the middle 50% of values. Anything outside `[Q1 - k×IQR, Q3 + k×IQR]` gets wiped out and turned
into a "missing value" (NaN), to be filled in properly by the imputer in the next step.

Important details:

- This runs **before** filling in missing values, and it runs on train, validation, **and** test
  data alike (only test/validation use the boundary numbers learned from training data — we never
  peek at validation/test to decide what "normal" looks like).
- This is different from removing a whole bad *row* (Section 3) — a single corrupted cell in an
  otherwise fine row shouldn't disqualify that whole row.
- We tried `k` (how wide the "normal" range is) from 2.0 to 4.0 and measured the downstream score.
  `k=2.5` won: it gave both a higher average score and a noticeably tighter (more consistent)
  spread of scores (0.0463 standard deviation) versus the more textbook-standard `k=3.0` (0.0688
  standard deviation). Tighter bounds caught more real corruption without falsely flagging normal
  values.
- We added this step because a whole-row anomaly detector (Section 3) is nearly blind to
  single-cell corruption — one broken number among 832 columns barely moves a row's overall
  "weirdness score."

---

## 3. Filling in the blanks (missing values)

**In kid terms:** Some boxes on the form are just empty — nobody wrote an answer. Instead of
leaving them blank (which most guessing machines can't handle) or making up something wild, we
write in the "typical" answer for that box, based on everyone else's answers.

**The real experiment:** We use `SimpleImputer(strategy="median", keep_empty_features=True)` —
median, not mean, because outliers haven't been removed yet at this point in the pipeline, and the
median doesn't get dragged around by extreme values the way an average does.

- We fit the "typical value per column" only on the training data, then use those same saved
  numbers to fill in validation and test data — we never let validation/test values leak into
  what "typical" means.
- `keep_empty_features=True` matters for a boring-but-important reason: if a column happened to be
  *completely* empty in training data, the imputer would otherwise silently delete that column,
  which shifts every column after it out of position and quietly breaks the whole pipeline.
- We also tried a fancier approach, `KNNImputer`, which fills in a blank by looking at other
  *similar* rows (using other features) rather than just the column average. It measured **worse**:
  0.4537 ± 0.0210 vs. plain median's 0.4656 ± 0.0688 (nested cross-validation R²). This makes sense
  given what we found in Section 0 — features barely relate to each other here, so "look at
  similar rows" doesn't have much real similarity to lean on, and just adds noise.

---

## 4. Throwing out whole rows that look "weird" overall

**In kid terms:** Now that every box on the form is filled in, look at each *whole form* (each
person's full set of answers) and ask: "does this person's whole pattern of answers look totally
different from everybody else's, in a way that's probably a mistake rather than a real person?"
If yes, we set that form aside — but only from the pile we study from, never from a form we're
actually supposed to grade.

**The real experiment:** We use `IsolationForest(contamination="auto")` — a method that doesn't
assume any particular bell-curve shape and doesn't need a full column-to-column relationship map
(which would be unreliable here, since we have almost as many features as rows). It flags roughly
4 out of 969 training rows as outliers and removes them.

- This step only ever touches the **training** rows — validation and test rows always get a
  prediction, no matter how strange they look, because in the real Kaggle test we're graded on
  every row.
- We tried forcing it to flag more rows (`contamination` = 0.01, 0.02, 0.05, 0.1) and compared
  downstream scores. `0.02` came out statistically tied with `"auto"` (0.492 vs. 0.492, well within
  a ~0.07 standard deviation), while every higher value scored worse and less consistently. So the
  default `"auto"` behavior was already close to the best choice — not something we forgot to tune.

---

## 5. Picking the best clues out of 832 (feature selection)

**In kid terms:** We have 832 clues, but many of them are either near-duplicates of each other
(basically saying the same thing twice) or just plain unhelpful noise. We want to keep a smaller,
smarter set of clues: no repeats, and only the ones that actually relate to age.

**The real experiment:** This happens in two steps.

### Step 5a — Remove near-duplicate clues

We compute the correlation between every pair of features. If two features correlate with each
other above `correlation_threshold=0.9`, we treat them as saying "the same thing" and only keep
one.

- The order we process features in matters: we go in **descending order of each feature's own
  correlation with age**, not the order columns happen to appear in the CSV. That way, out of two
  near-duplicate features, we keep whichever one is *actually more related to age*, instead of
  whichever one happened to be typed into the spreadsheet first (a pure accident of column order).

### Step 5b — Keep only the top 100 most age-related clues

Out of what's left after de-duplication, we score every feature with a univariate F-test
(`SelectKBest(f_regression)`) — basically "how cleanly does this one feature line up with age, on
its own?" — and keep only the top `k=100`.

### What we tried before landing here, and why it changed

This F-test approach **replaced** an earlier, fancier method called Boruta: for each feature, you
compare how useful a Random Forest thinks it is against a shuffled, fake ("shadow") copy of that
same feature, repeated over several trials, and only keep features that reliably beat their own
fake copy.

- Early on, Boruta looked good: requiring features to win in 100% of trials (`confirmation_threshold=1.0`)
  beat requiring only 50% of trials, which beat no feature selection at all (~0.49 vs. ~0.48 vs.
  ~0.41 R²).
- But once we found that SVR (Section 6) was our strongest model family, we re-ran the comparison
  specifically against SVR, across 3 different random data splits (seeds 42, 1, 7) to make sure we
  weren't just seeing a lucky split. Every single time, the simple F-test filter beat Boruta by a
  large margin: Boruta scored 0.5098 / 0.4875 / 0.4896 across the three seeds, while the F-test
  filter (k=100) scored 0.5290 / 0.5285 / 0.5147 — a 0.02 to 0.04 R² swing in favor of the F-test,
  every time.
- **Why:** Boruta's criterion is "does a tree ensemble find this useful," which rewards features
  that only help *in combination with others* (splits and interactions). SVR's RBF kernel instead
  rewards features with a clean, direct, one-on-one relationship to age — exactly what
  `f_regression` measures. Once SVR became our best model, we needed a feature selector that
  matched what SVR actually cares about.
- We swept `k` (how many features to keep) across `{75, 100, 125, 150}` — 100 was the consistent
  best (or tied-best) on every one of the 3 seeds.
- We also tried `mutual_info_regression` (a method that can catch non-linear relationships, not
  just straight-line ones). It scored consistently worse (~0.45–0.49) than both Boruta and the
  F-test — its extra flexibility doesn't pay off here, and it's a noisier estimator on a dataset
  with only a few hundred rows per fold, which actively hurts when used as a filter.
- We also tried using `Lasso` regression's own coefficients as a feature selector (keep whatever
  Lasso decides to give non-zero weight). This scored even worse (~0.43–0.46, peaking around
  `alpha=1.0`, which kept only ~16 features) — it's one step further removed from "does this
  feature actually correlate with age," since it depends on what a *penalized linear model*
  happens to zero out, not a direct measurement of relevance.

---

## 6. Making every clue "the same size" (scaling)

**In kid terms:** Imagine one clue is measured in millimeters and another in kilometers — a
guessing machine might think the kilometer clue matters way more just because the numbers are
bigger, even if that's not true. So we squash every clue onto the same, fair scale before the
guessing machine looks at them.

**The real experiment:** We use `QuantileTransformer(output_distribution="normal")` — this looks
at each feature's *rank* (is this value low, medium, or high compared to everyone else?) rather
than its raw distance from the average, and reshapes it into a nice bell curve.

- Earlier pipeline steps (outlier detection, Random Forest, Pearson correlation) don't care about
  scale at all, so this step couldn't have been done any earlier for free — but it matters a lot
  once we get to modeling, specifically for two model families: SVR and Ridge. (Tree-based models
  don't care about *any* monotonic rescaling of a single feature, so the choice of scaler never
  changes their scores.)
- We compared this against `RobustScaler` (which uses median and IQR instead of ranks) directly
  against SVR: `QuantileTransformer` scored 0.5104 ± 0.0664 vs. `RobustScaler`'s 0.4992 ± 0.0661,
  at the same `SVR(C=15, epsilon=3.0)` settings — a real, non-marginal gap. We also re-tuned SVR's
  `C`/`epsilon` from scratch under `QuantileTransformer` and landed on that exact same setting as
  the new best, confirming the win wasn't just a lucky match to settings chosen for the other
  scaler.
- One small technical note: `n_quantiles` is capped at the size of the training fold, since
  scikit-learn's default of 1000 is bigger than every fold we ever fit here (roughly 860–1200
  rows) — this just avoids a repeated warning, it doesn't change behavior.

---

## 7. Choosing the best guessing machine (model selection)

**In kid terms:** Now that the clues are clean, deduplicated, and fairly scaled, we let several
different "guessing machines" (models) each have a try at guessing ages, using only the practice
data (never peeking at the practice test we set aside in Section 1). Whichever machine guesses
best — fairly, on data it hasn't memorized — wins.

**The real experiment:** We test roughly 38 candidate models, grouped into 6 "families," each with
several different internal settings:

| Family | What it is, in kid terms | Settings we tried |
|---|---|---|
| **Ridge** | A straight-line guesser that's told not to trust any one clue too much | `alpha` (how cautious it is): 0.1, 1.0, 10.0, 100.0 |
| **RandomForest** | A big committee of simple yes/no decision trees, each only allowed to peek at a random handful of clues per question | `max_depth` (how many questions deep): 5, 10, unlimited; `min_samples_leaf` (smallest group size before it stops asking questions): 1, 5, 15 |
| **HistGradientBoosting** | A chain of trees where each new tree tries to fix the mistakes of the ones before it | `max_depth`: 3, 5, unlimited; learning rate (how big a correction each tree makes): 0.05, 0.1; `l2` (how cautious): 0.0, 1.0 |
| **CatBoost** | Another mistake-fixing chain of trees, tuned a bit differently | `depth`: 4, 6, 8; learning rate: 0.05, 0.1 |
| **SVR** | Draws a "tube" around the trend and only cares about points that fall outside the tube | `C` (how much it punishes points outside the tube): 10, 15, 20; `epsilon` (how wide the tube is): 2.0, 3.0 |
| **GPR (Gaussian Process)** | A guesser that also tells you *how sure* it is, and figures out its own best settings automatically instead of needing a grid search | One setting only (see below) |

### How we pick fairly: cross-validation, never on the practice test

We split the training rows into 5 folds (`KFold(shuffle=True, random_state=42)`), and for each
fold: train on the other 4/5, test on the held-out 1/5. Every candidate is scored this way, and the
winner is whoever has the best **average** score across all 5 folds — never based on the Section-1
practice test (`X_val`), which is only shown afterward as a diagnostic. Why not just pick based on
the practice test directly? Because with only ~243 rows in it, whichever model happens to fit
*that one particular slice's* noise best would win — not whichever model actually generalizes best.
We measured this: across different random splits, the same model's score on a single validation
slice swung between R² 0.42 and 0.54, just from which rows ended up in it.

**Feature selection and scaling are redone inside every single fold**, not once upfront — because
both of those steps look at the age labels (`y_train`). Doing them once outside the fold loop would
let information from each fold's "held-out" rows sneak into the features it's later tested on.
We measured the size of this cheating effect directly: ~0.014 R² of false optimism (a "leaky"
score of 0.500 vs. an honest, properly-nested score of 0.486, same settings otherwise).

### Model-family details worth knowing

- **RandomForest** needs `max_features="sqrt"` (only let each split look at a random subset of
  clues) — without it, trees can grow deep enough to almost perfectly memorize the training rows
  (train R² around 0.93), which doesn't help on new data at all.
- **HistGradientBoosting** needs `early_stopping=True` set explicitly — its own `"auto"` setting
  silently turns *off* early stopping below 10,000 rows (we only have ~900–1200), so without this
  it would run all 100 rounds regardless of overfitting.
- **CatBoost** runs with `verbose=False` (no wall of text) and no early stopping — its best,
  shallowest settings were already competitive without it.
- **SVR**'s `gamma="scale"` was fixed after testing — it consistently beat `"auto"` at every
  competitive `C` value. The `C`/`epsilon` grid is centered on a previously-measured peak:
  `C=15, epsilon=3.0` scored 0.4992 ± 0.0661 at the time, beating every tree-based candidate then
  available. We also tried epsilon values up to 8.0, and the score kept getting worse — confirming
  3.0 is a real peak, not just the edge of the grid we happened to test.
- **GPR**'s kernel (`ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)`)
  only needs *one* setting, unlike SVR's 6, because GPR automatically re-tunes its own length-scale
  and noise level during training by maximizing how likely the data is under its model — we
  confirmed this by trying a 3×3 grid of starting values (length_scale 5/10/20 × noise_level
  0.5/1.0/2.0) and got the *exact same* score (0.5367 ± 0.0940) every time, because the starting
  point gets optimized away regardless. Compared head-to-head against production
  `SVR(C=15, epsilon=2.0)` across 3 random-split seeds (42, 1, 7), GPR won every single time by a
  modest but consistent margin (0.5367/0.5339/0.5201 vs. SVR's 0.5332/0.5273/0.5182).

### Combining the best guessers into a team (stacking)

Rather than combine all ~38 candidates (most within a family are near-duplicates and would just
give a "team vote" that's really the same opinion repeated many times), we pick one champion per
family (best Ridge, best RandomForest, best HistGradientBoosting, best CatBoost, best SVR, best
GPR — 6 total), and blend their predictions with a simple linear combiner
(`LinearRegression(positive=True)`, so it can only ever take a non-negative-weighted average, never
subtract one model's opinion from another's).

- The blend's own score is computed just as honestly as any single model's — using the *same* 5
  fold boundaries, so no row's stacked prediction is ever influenced by a model that has seen that
  exact row.
- We deliberately keep the "team captain" (meta-learner) very simple, because blending models can
  overfit fast on a dataset this small (~969 training rows after outlier removal) — a fancy blender
  would just be memorizing which base model happened to be right on which practice rows.
- Adding CatBoost narrowed the stack's advantage down to "within noise" — a single RandomForest has
  occasionally beaten the stack on a given run since then. So stacking is no longer a guaranteed
  win, just historically the best-performing slot more often than not.

### The tie-breaking rule ("when in doubt, trust the team, not the star player")

If the single best individual model and the stacked team are statistically indistinguishable (the
average difference between them, fold by fold, is smaller than the natural noise in that
difference), we deploy the **stack** instead of the raw winner — even if the raw winner's number is
technically a hair higher.

- This is a known statistics idea (the "1 standard-error rule," used for example when pruning
  decision trees or picking Lasso's penalty strength): among options that are tied within noise,
  prefer the more stable one.
- **This isn't just theory — it fixed a real mistake we saw.** Two Kaggle submissions that only
  differed in one setting (5 vs. 10 cross-validation folds) picked *different* winners — the stack
  in one case, a plain RandomForest in the other — even though locally they were within 0.0006 to
  0.0015 of each other. The leaderboard score actually **dropped** (0.6344 → 0.6174) on the
  submission where the single model won the coin-flip instead of the stack. This rule exists to
  stop that from happening again.
- Importantly, this rule only ever looks at the training data's own cross-validation folds — it
  never peeks at the validation set or at Kaggle feedback, so it doesn't compromise the honesty of
  model selection, just its stability.

### One last, generous step: retrain on *everything* before predicting

Once the winning model (and its exact settings) is locked in using cross-validation on training
data only, we retrain that exact same recipe one more time — this time on training **and**
validation data combined — before it ever makes a prediction on the real test set. Model *choice*
never sees the validation data, but the model that actually produces our submission should use
every labeled row we have, not leave ~10% of it on the table. (For the stacked team, only the 6
champion models get this extra retraining — the blending weights stay exactly as learned from the
honest cross-validation, matching the same "train small, deploy big" pattern used one level down.)

### Things we tried and threw away (and why that's OK)

- **Balancing folds by age group (`StratifiedKFold` on age deciles):** we thought maybe one unlucky
  fold with a skewed age mix was hurting our score consistency. It actually made things *worse* —
  scores dropped and their spread roughly doubled (e.g., one RandomForest setting went from
  0.4903 ± 0.0680 unstratified to 0.4572 ± 0.1213 stratified). Turns out fold-to-fold difficulty
  here comes from which specific *people's* feature patterns land in which fold, not from their
  age — forcing age-balance doesn't fix that, it just produces a worse split by chance.
- **Letting RandomForest/boosting train on random subsets of rows (`max_samples`/`subsample` < 1):**
  measured as neutral — no real change to the team's score (0.5017 ± 0.0694 vs. 0.5026 ± 0.0703,
  within noise) — for 17 extra candidate settings, so we dropped it to save time.
- **ElasticNet and ExtraTrees as extra families:** both were redundant with what we already had.
  ElasticNet basically retraced Ridge's already-weak straight-line ceiling (~0.32–0.34). ExtraTrees
  always scored slightly worse than RandomForest on matching settings.
- **XGBoost:** removed for a structural reason, not a bad setting. Its scikit-learn interface needs
  an explicit "watch this held-out slice while training" (`eval_set`) to stop itself early —
  something our shared fold loop doesn't provide (unlike HistGradientBoosting, which carves out its
  own internal check automatically). Without that, it overfit badly at any real depth (train
  R²=1.0, cross-validation score collapsed to ~0.38–0.39), and even its safest, shallowest setting
  still lost to HistGradientBoosting's best.
- **10 folds instead of 5:** measured as statistically indistinguishable from 5 folds (well within
  the natural ~0.09–0.10 per-fold noise). We kept 5 for faster iteration while still tuning the
  candidate list — not a settled decision either way.

---

## Summary: the final recipe, start to finish

1. Split off 10% of the training data as an untouchable practice test.
2. Fix single broken numbers, column by column, using a range test (`k=2.5`) — applied to
   train, validation, and test.
3. Fill in missing values with each column's median, learned from training data only.
4. Remove ~4 whole training rows that look like overall mistakes (never touches validation/test).
5. Narrow 832 clues down to 100: drop near-duplicates (favoring the more age-relevant twin), then
   keep the top 100 by direct correlation with age.
6. Rescale every clue onto a fair, comparable, rank-based scale.
7. Let 6 different families of guessing machines compete honestly via 5-fold cross-validation,
   blend the best of each family into a team, and — if the team and the lone best model are tied
   within noise — trust the team. Retrain the winner one more time on all available labeled data
   before making real predictions.

Every one of these choices was arrived at by trying alternatives and measuring, not by guessing —
and several of the alternatives we tried (KNN imputation, Boruta, mutual information, Lasso
selection, RobustScaler, stratified folds, XGBoost, extra sampling axes) turned out worse and were
deliberately left out, which is just as important a finding as what we kept.
