# Machine Failure Prediction — DATA DIVERGENCE, Track 1 (Beginner)

**Team: Gradient Descenders** — Dharunesh A & Valliappan.L

Predict whether a machine fails, from 5 sensor readings and its `Type`.

---

## Results

5-fold cross-validated, out-of-fold, decision threshold 0.50:

| Metric | Value |
|---|---|
| **PR-AUC** | **0.9309** |
| **F1** | **0.9142** |
| Precision | 0.9957 |
| Recall | 0.8450 |

Confusion matrix over all 7,999 rows:

|  | predicted OK | predicted FAIL |
|---|---|---|
| **actually OK** | 7727 | 1 |
| **actually FAIL** | 42 | 229 |

**229 of 271 real failures caught, with 1 false alarm.**

---

## Approach

The raw sensor readings carry almost no signal — a model trained on them directly scores PR-AUC
0.41. The data was generated from physical failure rules, so we derived five features encoding
those rules (`hdf`, `pwf`, `osf`, `wear_pos`, `is_L`) and fit a plain logistic regression on top.
That took PR-AUC from **0.41 → 0.93**, and logistic regression beat all 11 other models we tried.

Reasoning, figures and validation: **[`Methodology_Track1.pdf`](Methodology_Track1.pdf)**.

---

## Run it

**Colab** — upload the notebook and `train.csv`, then `Runtime → Run all` (~1 min).

**Locally:**

```bash
pip install -r requirements.txt
python verify.py
```

`verify.py` runs the notebook and **prints the results**, so you can check them without opening
anything. It exits non-zero if a cell fails. Use `--skip-run` to re-print results already saved.

**Score the hidden test file:**

```bash
python predict.py test.csv submission.csv
```

---

## Files

| File | Purpose |
|---|---|
| `Track1_Machine_Failure_Prediction.ipynb` | Full notebook — EDA, features, training, validation |
| `Methodology_Track1.pdf` | 2-page methodology summary |
| `predict.py` | Inference on a new CSV |
| `model.joblib` | Trained pipeline + threshold |
| `verify.py` | Runs the notebook and prints the results |
| `train.csv` | Provided dataset (7,999 rows, 0 missing values) |
| `requirements.txt` | Dependencies |

---

## Requirements

Python 3.9+. Verified end-to-end on Python 3.12 with pandas 3.0.5, scikit-learn 1.9.1,
numpy 1.26.4, matplotlib 3.11.2, joblib 1.6.0.
