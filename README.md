# Machine Failure Prediction — DATA DIVERGENCE, Track 1 (Beginner)

**Team: Gradient Descenders** — Dharunesh A & Valliappan.L

Predict whether a machine fails, from 5 sensor readings and its `Type`.

---

## Results

Out-of-fold over 25 seeds × 5 folds (125 fits), decision threshold 0.50:

| Metric | Value |
|---|---|
| **PR-AUC** | **0.9305** |
| **F1** | **0.9160** |
| Precision | 1.0000 |
| Recall | 0.8450 |

Confusion matrix over all 7,999 rows:

|  | predicted OK | predicted FAIL |
|---|---|---|
| **actually OK** | 7728 | 0 |
| **actually FAIL** | 42 | 229 |

**229 of 271 real failures caught, with 0 false alarms.**

The cut-off is the untuned default of 0.50 — it was never adjusted.

---

## Approach

The raw sensor readings carry almost no signal — a model trained on them directly scores PR-AUC
0.41. The data was generated from physical failure rules, so we derived five features encoding
those rules (`hdf`, `pwf`, `osf`, `wear_pos`, `is_L`) and fit a plain logistic regression on top.
That took PR-AUC from **0.41 → 0.93**, and logistic regression beat all 11 other models we tried.

Reasoning, figures and validation: **[`Methodology_Track1.pdf`](Methodology_Track1.pdf)**.

---

## Run it locally

Full sequence, starting from nothing. Copy the whole block:

```bash
git clone https://github.com/<your-username>/Track1-GradientDescenders.git
cd Track1-GradientDescenders
pip install -r requirements.txt
python verify.py
```

Replace `<your-username>` with your GitHub username. On Windows, use **Command Prompt** or
**PowerShell** — the commands are identical, just type each one on its own line.

`verify.py` runs the notebook and **prints the results** (PR-AUC, F1, precision, recall, confusion
matrix, model ranking, overfitting checks), so you can confirm them without opening anything. It
takes about two minutes and exits non-zero if a cell fails.

Once it has run once, `python verify.py --skip-run` re-prints saved results without re-running.

**Score the hidden test file:**

```bash
python predict.py test.csv submission.csv
```

---

## Run it in Colab

No install needed. Upload `Track1_Machine_Failure_Prediction.ipynb` and `train.csv` to the same
Colab folder (folder icon on the left, drag the files in), then `Runtime → Run all` (~2 min).

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
