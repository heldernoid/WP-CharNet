"""Everything needed to write one letter's referees (TRAINING data only): for each class given (case-sensitive),
its accuracy, its top thieves, and for each thief the features that best separate the misread images from the
thief's correctly read images, with three means: misread letter / correctly read letter / thief. A referee should
only use features where both letter means agree (a feature the correctly read letters also have pushes them down).
Usage: plan.py <classes> [k] [n]"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eval.byclass as B  # noqa: E402

if __name__ == "__main__":
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    y, a, ok, SL, S, F, names = B.answers("study")
    names = np.array(names)
    for c in sys.argv[1]:
        mine = y == c
        th = Counter(a[mine & ~ok]).most_common(k)
        print(f"\n## {c}: {100 * ok[mine].mean():.0f}%   thieves: " + " ".join(f"{p}:{m}" for p, m in th))
        for p, m in th:
            A = F[mine & (a == p)]
            Bm = (y == p) & ok
            if Bm.sum() < 10:
                Bm = (np.char.upper(y) == p.upper()) & ok
            Bf = F[Bm]
            sep = np.abs(A.mean(0) - Bf.mean(0)) / (np.sqrt((A.var(0) + Bf.var(0)) / 2) + 0.05)
            o = np.argsort(-sep)[:n]
            G = F[mine & ok]
            # rank by separation, but only features where the correctly read letter sides with the misread one
            agree = np.abs(G.mean(0) - A.mean(0)) < np.abs(G.mean(0) - Bf.mean(0))
            o = [i for i in np.argsort(-sep) if agree[i]][:n]
            print(f"  vs {p} ({m}): " + "  ".join(f"{names[i]} {A[:, i].mean():.2f}/{G[:, i].mean():.2f}/{Bf[:, i].mean():.2f}" for i in o))
