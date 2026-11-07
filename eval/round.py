"""One line per round of changes, TRAINING splits only: digits and letters together, so neither side
quietly gets worse.  Usage: round.py [--holdout]   (--holdout: the unstudied chunks, once per round)

  MNIST digits   train[0:6000]  (holdout train[6000:9000])  digits-only, and all 62 (must name the digit)
  EMNIST letters train[0:10400] (holdout train[60000:65200]) letters-only, and all 62, case-folded
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eval.letters as LT  # noqa: E402
from charnet.engine import load  # noqa: E402
from eval.mnist import read_idx  # noqa: E402

_net = None
# Shapes no rule can tell apart without context: a stick, a ring, a Z-shape, an S-shape, a loop on a stem.
LOOKALIKE = [set("0Oo"), set("1Il"), set("2Zz"), set("5Ss"), set("9gq")]


def same(a, b):
    """Answer a fits label b: equal case-folded, or in one look-alike group with either case of b
    (an EMNIST Letters label L stands for L or l, so a stick read as 1 is a look-alike)."""
    return a.upper() == b.upper() or any(a in g and (b.upper() in g or b.lower() in g) for g in LOOKALIKE)


def _init():
    global _net
    _net = load(LT.ARCH, LT.EDITS)


def _digit(img):
    T = _net.forward(img.astype(float) / 255.0)
    return [T["cls:" + c] for c in _net.arch["classes"]] + [T[f"dig:{d}"] for d in range(10)]


if __name__ == "__main__":
    hold = "--holdout" in sys.argv
    X = read_idx("train-images-idx3-ubyte.gz")
    y = read_idx("train-labels-idx1-ubyte.gz")
    r = range(6000, 9000) if hold else range(0, 6000)
    with Pool(8, _init) as p:
        SS = np.array(p.map(_digit, X[r], chunksize=50))
    S, SD = SS[:, :62], SS[:, 62:]
    t = y[r]
    d10 = (SD.argmax(1) == t).mean()
    d62 = (S.argmax(1) == t).mean()
    C = np.array(_init() or _net.arch["classes"])
    dlk = np.mean([same(p, str(q)) for p, q in zip(C[S.argmax(1)], t)])
    Sl, _, classes, _ = LT.run("holdout" if hold else "study")
    _, Yl = LT.data()
    a62, a52 = LT.answers(Sl, classes, LT.let_scores("holdout" if hold else "study"))
    llk = np.mean([same(p, q) for p, q in zip(np.array(classes)[Sl.argmax(1)], Yl[LT.CHUNKS["holdout" if hold else "study"]])])
    yl = Yl[LT.CHUNKS["holdout" if hold else "study"]]
    print(f"{'HOLDOUT' if hold else 'study'}: digits {100 * d10:.1f}% (all 62: {100 * d62:.1f}%)   "
          f"letters {100 * (a52 == yl).mean():.1f}% (all 62: {100 * (a62 == yl).mean():.1f}%)   "
          f"| all 62, look-alikes merged: digits {100 * dlk:.1f}%, letters {100 * llk:.1f}%")
