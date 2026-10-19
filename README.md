# Predicting Machine Failure — DATA DIVERGENCE, Track 1 (Beginner)

**Team: Gradient Descenders** — Dharunesh A & Valliappan S

Binary classification of `Machine failure` from five sensor readings plus product type.
Scored on **PR-AUC** and **F1** for the minority class against a hidden `test.csv` with an
identical schema.

---

## Headline results

All numbers are **out-of-fold**: every row is scored by a model that never saw it during
training. Averaged over 5 stratified 5-fold splits.

| Metric | Score | What it means |
|---|---|---|
| **PR-AUC** | **0.931** | Random guessing scores 0.034 |
| **F1 @ 0.5** | **0.914** | The two judging metrics |
| **Precision** | **99.6%** | 1 false alarm in 7,728 healthy rows |
| **Recall** | **84.5%** | 229 of 271 real failures caught |
| Confusion matrix | `[[7727, 1], [42, 229]]` | out-of-fold, cut-off 0.5 |

> **On the one false alarm.** This is not a rounding artefact. One borderline healthy row
> genuinely moves depending on how many splits are averaged: at 5 splits it is flagged
> (precision 99.6%, F1 0.914); average 20 splits instead and it falls below the cut-off
> (precision 100%, F1 0.916). We report the lower pair so the headline does not rest on a
> choice we made.

---

## The idea in one paragraph

The raw readings barely separate healthy rows from failing ones — their averages are almost
identical. But the dataset was generated from **physical failure rules**, and once those rules
are written down as features, the problem becomes close to linearly separable. Three rules are
*exact*: every row they flag is a real failure. A fourth, tool wear, is genuinely
probabilistic. We turn the three exact rules plus a graded wear term plus product type into
five features, and fit a plain logistic regression.

**Features mattered far more than model choice.** With raw readings, logistic regression scores
PR-AUC 0.41. With the five features it scores 0.93 — and beats all twelve alternative model
families we tried (best rival: a neural network at 0.929).

| Setup | PR-AUC |
|---|---|
| Raw readings + logistic regression | 0.407 |
| Raw readings + gradient boosting | 0.790 |
| 5 features + gradient boosting | 0.924 |
| **5 features + logistic regression** | **0.931** |

---

## The five features

| Feature | Plain name | Definition |
|---|---|---|
| `hdf` | heat dissipation | `(process temp − air temp) < 8.6 K` **and** `rpm < 1380` |
| `pwf` | power failure | `torque × rpm × 2π/60` outside 3500–9000 W |
| `osf` | overstrain | `wear × torque` over the per-type limit (L 11000, M 12000, H 13000) |
| `wear_pos` | tool wear | position within the 200–239 min band, scaled 0→1 |
| `is_L` | product type | 1 for type L |

Rule purity, measured on `train.csv`: `hdf` flags 94 rows (100% real failures), `pwf` 73 (100%),
`osf` 77 (100%). Together they catch **229 of 271 failures with zero false alarms**. The wear
band holds 624 rows, of which only ~16% fail — so there is no sharp line to find there.

### ⚠️ Do not round the temperature gap

103 rows have a temperature gap of exactly 8.6 K in decimal arithmetic, but float64 stores
`310.9 − 302.3` as `8.599999999999966` or `8.600000000000023` depending on the operand pair —
and the data generator compared that **raw binary value** against 8.6.

This is not an error to clean up; it is what the labels were built from. Rounding to the exact
decimal loses **11 real failures** (94 → 83 caught, PR-AUC 0.931 → 0.898). Using
`round(gap, 6) <= 8.6` instead adds 10 false alarms. Section 3 of the notebook prints all three
variants side by side so you can check this yourself.

---

## Files

| File | Purpose |
|---|---|
| `Track1_Machine_Failure_Prediction.ipynb` | The full, executed notebook — EDA, features, training, validation, stress tests |
| `verify.py` | Runs the notebook and **prints the results** so you can check them without opening anything |
| `Methodology_Track1.pdf` | 2-page methodology summary (figures A–F) |
| `predict.py` | Standalone inference — `python predict.py test.csv submission.csv` |
| `model.joblib` | Trained pipeline + threshold, loaded by `predict.py` |
| `train.csv` | The provided dataset (7,999 rows, 0 missing values — no cleaning required) |
| `requirements.txt` | Dependencies |

---

## How to reproduce

**Option A — Google Colab (easiest).**

1. Upload `Track1_Machine_Failure_Prediction.ipynb` and `train.csv` to the same Colab folder
   (folder icon on the left, drag the files in).
2. `Runtime → Run all`.
3. Takes about one minute. All outputs, tables and figures are already saved in the file.

**Option B — locally (recommended, because it prints the results).**

```bash
pip install -r requirements.txt
python verify.py
```

That runs the whole notebook and then prints the headline numbers to your terminal, so
you can check them without opening anything:

```
15 code cells executed (+ 22 explanation sections = 37 total), 0 errors, 3 figures

HEADLINE  (5-fold out-of-fold, decision threshold 0.50)
  PR-AUC     0.9309
  F1         0.9142
  Precision  0.9957   <- when it raises an alarm, it is right this often
  Recall     0.8450   <- share of real failures it catches
  ...
  -> 229 of 271 real failures caught, 1 false alarm
```

It also prints the 12-model ranking, the overfitting checks and the inference checks, and
**exits with a non-zero status if any cell failed** — so "command finished" means it worked.

Once the notebook has been run once, use `python verify.py --skip-run` to just re-print the
results from the saved file without waiting for it to run again.

<details>
<summary>Raw nbconvert command (if you prefer it)</summary>

```bash
jupyter nbconvert --to notebook --execute --inplace Track1_Machine_Failure_Prediction.ipynb
```

One line, no backslash. Note this **prints nothing readable** — it writes the results into
the `.ipynb` file, so you must open the notebook to see them. That is why `verify.py` exists.

</details>

**Option C — VS Code (no terminal needed).** Install the *Jupyter* extension, open
`Track1_Machine_Failure_Prediction.ipynb`, then **Run All**.

*Verified end-to-end on Python 3.12 with pandas 3.0.5, scikit-learn 1.9.1, numpy 1.26.4,
matplotlib 3.11.2 and joblib 1.6.0: 37 cells, 0 errors, identical results.*

**Scoring the hidden test set:**

```bash
python predict.py test.csv submission.csv
```

`predict.py` expects the same columns as `train.csv`. It is schema-robust — verified to work
with the target column missing, with extra columns (`UDI`, `Product ID`) present, with columns
in a different order, and on a single row. All four checks run in section 9 of the notebook.

You can also call it from Python:

```python
from predict import predict
labels, probabilities = predict(df)
```

---

## Validation, and why we believe the number

* **Stratified 5-fold cross-validation**, out-of-fold scoring, 5 random splits averaged.
* **Nested CV** — re-choosing the regularisation strength *inside* every fold still gives
  PR-AUC 0.9304 ± 0.0209, so the tuning did not leak into the score.
* **Train-small / test-big** — training on 10% of the rows still gives 0.931.
* **Swapped halves** — 0.931 ± 0.001 over 20 random halvings.
* **Model zoo** — twelve standard families on identical features; none beats logistic
  regression.
* **Threshold stability** — F1 is flat from a cut-off of 0.5 to 0.7, so the shipped 0.5 is
  optimal and needs no tuning.

The score does not move when the data is cut differently, because the knowledge sits in the
features, not in the model.

---

## Honest limitations

* Every number above comes from splitting `train.csv` itself, and the rule thresholds
  (8.6 K, 1380 rpm, 3500/9000 W, per-type limits) were also found in that same file. So the
  honest claim is narrow: **this model works on machines like this one** — the same L, M and H
  product types, run the same way.
* A machine with genuinely different characteristics is a different problem. Those cut-offs are
  physical constants of this design; a different spindle, tool or duty cycle shifts every one.
  The five features would still be the right idea, but each threshold needs re-fitting on the
  new machine's own data before anyone acted on it.
* On the hidden test set — same columns, so we assume the same machine — we expect PR-AUC around
  0.93 with F1 nearer 0.90–0.91.
* On this data, recall above about 0.97 looks unreachable: some failures violate no rule at
  all. Flagging every rule *plus* the entire wear band still only reaches recall 0.974, and
  that costs 522 false alarms.

---

## Decisions we would defend

* **Logistic regression over gradient boosting.** It scores highest *and* its coefficients can
  be printed and argued with (`hdf` +0.94, `pwf` +0.82, `osf` +0.77, `wear_pos` +0.54,
  `is_L` −0.08). The three rule features carrying the largest weights is evidence the model
  picked up the physics rather than noise.
* **No class weighting.** `class_weight="balanced"` looks like the obvious fix for rare
  failures, but it moves the best cut-off to ~0.95 — and if a plain 0.5 is then applied, F1
  collapses from 0.914 to 0.60. We kept honest probabilities so 0.5 means what it says.
* **A 0.5 cut-off.** F1 is maximised there, and zero-or-one false alarms is what keeps a
  maintenance team trusting the system. Section 7 shows the full trade-off table for factories
  with a different cost structure.
