"""MNIST TRAINING split only: which named features tell digit a from digit b?

Usage: contrast.py a b [k] [prefix,...] [--err]
  default : all a images vs all b images in train[0:6000]
  --err   : a images the network reads as b vs b images it reads correctly
            (what a referee for a/b has to tell apart)
Ranks features by separation |mean_a - mean_b| / pooled std.
Features and predictions are cached in results/train_feats.npz."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.study import X, Y, NET  # noqa: E402

NAMES = NET.L["L7"]["out"] + NET.L["L8"]["out"]
CACHE = os.path.join(ROOT, "results", "train_feats.npz")
EDITS = os.path.join(ROOT, "weights", "charnet.edits")


def feats(n=6000):
    stamp = os.path.getmtime(EDITS)
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if float(z["stamp"]) == stamp and list(z["names"]) == NAMES and "pred" in z:
            return z["F"], z["pred"]
    F, P = [], []
    for i in range(n):
        T = NET.forward(X[i] / 255.0)
        F.append([T[k] for k in NAMES])
        P.append(int(np.argmax([T[f"cls:{d}"] for d in range(10)])))
    F, P = np.array(F), np.array(P)
    np.savez(CACHE, F=F, pred=P, names=np.array(NAMES), stamp=stamp)
    return F, P


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--err"]
    err = "--err" in sys.argv
    a, b = int(args[0]), int(args[1])
    k = int(args[2]) if len(args) > 2 else 12
    pre = tuple(args[3].split(",")) if len(args) > 3 else None
    F, P = feats()
    y = Y[:len(F)]
    ma, mb = (y == a, y == b) if not err else ((y == a) & (P == b), (y == b) & (P == b))
    A, B = F[ma].mean(0), F[mb].mean(0)
    sd = np.sqrt((F[ma].var(0) + F[mb].var(0)) / 2) + 0.05
    sep = np.abs(A - B) / sd
    if pre:
        sep = np.where([n.startswith(pre) for n in NAMES], sep, -1)
    order = np.argsort(-sep)[:k]
    tag = f"{a}->{b} errors ({ma.sum()}) vs {b} ok ({mb.sum()})" if err else f"{a} vs {b}"
    print(f"{tag}: " + "  ".join(f"{NAMES[i]} {A[i]:.2f}/{B[i]:.2f}({sep[i]:.1f})" for i in order))
