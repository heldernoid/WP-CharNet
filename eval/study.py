"""Study tools for MNIST digits: TRAINING split only. Never call these on a test split.

  sheet <out.png> <digit> [n]            random training images of one digit
  errors <out.png> <true> <pred> [n]     training images of <true> that the network reads as <pred>
  confusion [n]                          digits-only confusion on the first n training images
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from charnet.engine import load  # noqa: E402
from eval.mnist import read_idx  # noqa: E402

X = read_idx("train-images-idx3-ubyte.gz")
Y = read_idx("train-labels-idx1-ubyte.gz")
NET = load(os.path.join(ROOT, "arch.json"), os.path.join(ROOT, "weights", "charnet.edits"))
HOLDOUT = range(6000, 9000)          # never viewed: the unstudied check chunk
STUDY = np.array([i for i in range(len(Y)) if i not in HOLDOUT])


def digit_pred(img):
    T = NET.forward(img.astype(float) / 255.0)
    return int(np.argmax([T[f"cls:{d}"] for d in range(10)]))


def save_sheet(imgs, out, cols=20):
    imgs = list(imgs) or [np.zeros((28, 28))]
    rows = (len(imgs) + cols - 1) // cols
    S = np.full((rows * 30, cols * 30), 80, np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        S[r * 30 + 1:r * 30 + 29, c * 30 + 1:c * 30 + 29] = im
    Image.fromarray(S).resize((S.shape[1] * 2, S.shape[0] * 2), Image.NEAREST).save(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    rng = np.random.default_rng(0)
    if cmd == "sheet":
        out, d = sys.argv[2], int(sys.argv[3])
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
        idx = rng.choice(STUDY[Y[STUDY] == d], n, replace=False)
        save_sheet(X[idx], out)
    elif cmd == "errors":
        out, t, p = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 40
        found = []
        for i in rng.permutation(STUDY[Y[STUDY] == t]):
            if digit_pred(X[i]) == p:
                found.append(i)
                if len(found) == n:
                    break
        print("indices:", found[:10], "...")
        save_sheet(X[found], out)
    elif cmd == "confusion":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        C = np.zeros((10, 10), int)
        for i in range(start, start + n):
            C[Y[i], digit_pred(X[i])] += 1
        print("train[%d:%d] digits-only acc %.1f%%" % (start, start + n, 100 * np.trace(C) / n))
        print("per digit:", " ".join(f"{d}:{100 * C[d, d] / max(1, C[d].sum()):.0f}%" for d in range(10)))
        print(C)
