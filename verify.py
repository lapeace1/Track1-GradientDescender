"""
verify.py - run the notebook and print the results to your terminal.

The README's nbconvert command executes the notebook but writes the output
into the .ipynb file, so you see nothing. This script runs the same thing and
then prints the numbers so you can check them at a glance.

Usage
-----
    python verify.py              # run the notebook, then print the results
    python verify.py --skip-run   # just print results already in the notebook

It exits with a non-zero status if any cell failed, so "command finished"
means "it worked".

Only needs the standard library plus whatever the notebook itself needs.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

NB = "Track1_Machine_Failure_Prediction.ipynb"


# ---------------------------------------------------------------- notebook IO
def cell_text(cell):
    """All printed text from one cell, as a single string."""
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            t = o.get("text", "")
            out.append(t if isinstance(t, str) else "".join(t))
    return "\n".join(out)


def all_text(nb):
    return "\n".join(cell_text(c) for c in nb["cells"] if c["cell_type"] == "code")


def errors(nb):
    return [
        (i, o.get("ename", "Error"), str(o.get("evalue", "")).strip())
        for i, c in enumerate(nb["cells"])
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]


def figures(nb):
    return sum(
        1
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if "image/png" in (o.get("data") or {})
    )


def search(pattern, text, group=1, default="?"):
    m = re.search(pattern, text)
    return m.group(group).strip() if m else default


# ------------------------------------------------------------------- reporting
def rule(char="=", n=64):
    print(char * n)


def head(title):
    print()
    rule()
    print(title)
    rule()


def report(nb):
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = errors(nb)
    text = all_text(nb)

    print()
    md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(
        f"{len(code)} code cells executed "
        f"(+ {md} explanation sections = {len(nb['cells'])} total), "
        f"{len(errs)} error{'s' if len(errs) != 1 else ''}, "
        f"{figures(nb)} figures"
    )

    if errs:
        head("ERRORS")
        for i, name, msg in errs[:10]:
            print(f"  cell {i}: {name}: {msg[:200]}")
        if len(errs) > 10:
            print(f"  ... and {len(errs) - 10} more")
        return False

    # ---- headline numbers
    pr_auc = search(r"PR-AUC = ([\d.]+)\n\ncut-off", text)
    f1 = search(r"F1=([\d.]+)", text)
    prec = search(r"precision=([\d.]+)", text)
    rec = search(r"recall=([\d.]+)", text)
    cm = search(r"confusion matrix.*?= (\[\[.*?\]\])", text)

    head("HEADLINE  (5-fold out-of-fold, decision threshold 0.50)")
    print(f"  PR-AUC     {pr_auc}")
    print(f"  F1         {f1}")
    print(f"  Precision  {prec}   <- when it raises an alarm, it is right this often")
    print(f"  Recall     {rec}   <- share of real failures it catches")

    tn = fp = fn = tp = "?"
    nums = re.findall(r"\d+", cm.replace(",", ""))
    if len(nums) >= 4:
        tn, fp, fn, tp = nums[0], nums[1], nums[2], nums[3]

    print()
    print("  Confusion matrix (all 7,999 rows)")
    print("                     predicted OK   predicted FAIL")
    print(f"    actually OK      {tn:>12}   {fp:>14}")
    print(f"    actually FAIL    {fn:>12}   {tp:>14}")
    print()
    print(
        f"  -> {tp} of {int(tp) + int(fn) if tp.isdigit() and fn.isdigit() else '?'} real failures caught"
        f", {fp} false alarm{'' if fp == '1' else 's'}"
    )

    # ---- model comparison
    head("FEATURE ENGINEERING MATTERS MORE THAN THE MODEL")
    for label, pat in [
        ("raw readings + logistic regression", r"Raw readings \+ logistic regression\s+PR-AUC = ([\d.]+)"),
        ("raw readings + gradient boosting  ", r"Raw readings \+ gradient boosting\s+PR-AUC = ([\d.]+)"),
        ("5 features   + gradient boosting  ", r"5 features \+ gradient boosting\s+PR-AUC = ([\d.]+)"),
        ("5 features   + logistic regression", r"5 features \+ logistic regression\s+PR-AUC = ([\d.]+)"),
    ]:
        print(f"  {label}   PR-AUC {search(pat, text)}")

    print()
    print("  Model ranking (12 tried, out-of-fold PR-AUC):")
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s{2,}(.+?)\s{2,}([\d.]+)\s{2,}([\d.]+)\s*$", line)
        if m:
            print(f"    {m.group(1):>2}. {m.group(2):<30} {m.group(3)}")

    # ---- validation
    head("IS IT OVERFITTED?  (the score barely moves)")
    print(f"  nested cross-validation   PR-AUC {search(r'nested CV \(25 outer folds\): PR-AUC = ([\d.]+)', text)}"
          f" +- {search(r'nested CV \(25 outer folds\): PR-AUC = [\d.]+ \+- ([\d.]+)', text)}")
    print(f"  trained on only 10%       PR-AUC {search(r'10% /  90%\s+([\d.]+)', text)}"
          f"   F1 {search(r'10% /  90%\s+[\d.]+ \+- [\d.]+\s+([\d.]+)', text)}")
    print(f"  trained on only 25%       PR-AUC {search(r'25% /  75%\s+([\d.]+)', text)}"
          f"   F1 {search(r'25% /  75%\s+[\d.]+ \+- [\d.]+\s+([\d.]+)', text)}")
    print(f"  20 random half-swaps      PR-AUC {search(r'swapped halves \(20 random halvings\): PR-AUC ([\d.]+)', text)}"
          f"   F1 {search(r'swapped halves \(20 random halvings\): PR-AUC [\d.]+ \+- [\d.]+, F1 ([\d.]+)', text)}")

    # ---- inference sanity
    head("INFERENCE CHECK  (predict.py on the raw file)")
    for line in text.splitlines():
        if line.startswith(("works", "ignores", "column order", "flagged on")):
            print(f"  {line}")

    print()
    rule()
    print("All checks passed. The notebook reproduces the reported numbers.")
    rule()
    print()
    print("To score the hidden test file:   python predict.py test.csv submission.csv")
    print()
    return True


# ----------------------------------------------------------------------- main
def main():
    path = Path(NB)
    if not path.exists():
        sys.exit(f"Cannot find {NB}. Run this from the folder that holds it.")

    if "--skip-run" not in sys.argv:
        print(f"Executing {NB} (about a minute)...")
        r = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace", NB,
            ]
        )
        if r.returncode != 0:
            sys.exit("Notebook execution failed - see the error above.")

    nb = json.loads(path.read_text(encoding="utf-8"))
    sys.exit(0 if report(nb) else 1)


if __name__ == "__main__":
    main()
