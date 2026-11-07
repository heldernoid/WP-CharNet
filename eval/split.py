"""Split a class's letters-only score into its best style and the referee part, on its misread and correctly read
study images; list the referees that push it down most. TRAINING data only.  Usage: split.py <c>"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eval.byclass as B  # noqa: E402
from charnet.engine import load  # noqa: E402

if __name__ == "__main__":
    c = sys.argv[1]
    y, a, ok, SL, S, F, names = B.answers("study")
    X = B.sets()["study_X"]
    net = load(B.ARCH, B.EDITS)
    for label, mask in (("misread", (y == c) & ~ok), ("read right", (y == c) & ok)):
        js = np.nonzero(mask)[0][:80]
        best, let, wbest, wlet, push = [], [], [], [], Counter()
        for j in js:
            T = net.forward(B.prep(X[j]))
            best.append(T["best:" + c]); let.append(T["let:" + c])
            w = a[j] if label == "misread" else max((k for k in B.LETTERS if k.upper() != c.upper()), key=lambda k: T["let:" + k])
            wbest.append(T["best:" + w]); wlet.append(T["let:" + w])
            for name in net.L["L13S"]["out"]:
                if name.startswith("say:"):
                    x, z = name[4:].split(">")
                    if z == c and not x.isdigit() and T[name] > 0.01 and T[name] >= T["losk:" + c] - 1e-9:
                        push[name[4:]] += 1.5 * T[name] / len(js)
        print(f"{c} {label} ({len(js)}): own best style {np.mean(best):.2f} -> after referees {np.mean(let):.2f}   |   "
              f"winner/runner-up best style {np.mean(wbest):.2f} -> after referees {np.mean(wlet):.2f}")
        print("   pushed down by: " + "  ".join(f"{k} {v:.2f}" for k, v in push.most_common(8)))
