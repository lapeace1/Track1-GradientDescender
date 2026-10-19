import numpy as np, pandas as pd, joblib
TARGET = "Machine failure"

def feats_min(d):
    d = d.drop(columns=[TARGET], errors="ignore")
    td = d["Process temperature [K]"] - d["Air temperature [K]"]        # NOT rounded
    pw = d["Torque [Nm]"] * d["Rotational speed [rpm]"] * 2 * np.pi / 60
    ot = d["Tool wear [min]"] * d["Torque [Nm]"]
    lim = d["Type"].map({"L": 11000, "M": 12000, "H": 13000})
    wear = d["Tool wear [min]"]
    return pd.DataFrame({
        "hdf": ((td < 8.6) & (d["Rotational speed [rpm]"] < 1380)).astype(int),
        "pwf": ((pw < 3500) | (pw > 9000)).astype(int),
        "osf": (ot > lim).astype(int),
        "wear_pos": ((wear - 200) / 40).clip(0, 1) * wear.between(200, 239),
        "is_L": (d["Type"] == "L").astype(int),
    })

def predict(data, threshold=None):
    # data: DataFrame with the same columns as train.csv. Returns (labels, probabilities).
    art = joblib.load("model.joblib")
    prob = art["model"].predict_proba(feats_min(data))[:, 1]
    thr = art["threshold"] if threshold is None else threshold
    return (prob >= thr).astype(int), prob

if __name__ == "__main__":
    import sys
    labels, probs = predict(pd.read_csv(sys.argv[1]))
    pd.DataFrame({"Machine failure": labels,
                  "failure_probability": probs}).to_csv(sys.argv[2], index=False)
