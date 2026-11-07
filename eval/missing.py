"""For a class (case-sensitive), on its misread study images: the best style's EXPECTED parts and how often each
is actually present (mean value), plus the score margin to the winner. A part that is usually missing is either
detected badly or is not how people write it. TRAINING data only.  Usage: missing.py <c>"""
import os
import sys

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
    Wl, Ll = net.P["L9"]["W"], net.L["L9"]
    styles = net.L["L10"]["groups"]["best:" + c]
    for label, mask in (("misread", (y == c) & ~ok), ("read right", (y == c) & ok)):
        js = np.nonzero(mask)[0][:80]
        vals, margin, won = {}, [], {}
        for j in js:
            T = net.forward(B.prep(X[j]))
            st = max(styles, key=lambda k: T[k])
            won[st] = won.get(st, 0) + 1
            o = Ll["out"].index(st)
            for i in np.nonzero(Wl[o])[0]:
                if Wl[o, i] > 0:
                    vals.setdefault((st, Ll["in"][i], round(float(Wl[o, i]), 3)), []).append(T[Ll["in"][i]])
            margin.append(T["let:" + c] - max(T["let:" + k] for k in B.LETTERS if k.upper() != c.upper()))
        print(f"{c} {label} ({len(js)}): styles {won}, mean margin to the best other letter {np.mean(margin):+.2f}")
        for st in won:
            parts = sorted(((n, w, np.mean(v)) for (s_, n, w), v in vals.items() if s_ == st), key=lambda t: t[2])
            print(f"   {st}: " + "  ".join(f"{n}(w{w:.2f}) {m:.2f}" for n, w, m in parts))
