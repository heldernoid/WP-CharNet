"""Letter-by-letter testing on EMNIST ByClass TRAINING data (upper and lower case kept apart). Never a test split.

Sets (balanced, letters only, built once into results/byclass_sets.npz):
  study    first 300 images of every letter class in train[0:120000]      (viewed; decisions made here)
  holdout  first 120 images of every letter class in train[400000:440000] (never viewed; checked when a letter is done)
Accuracy = letters-only view (let:* outputs), case-folded: answering 'c' for a C counts as right, since the
normaliser removes size and baseline and many capitals and small letters then look the same.

Usage:
  byclass.py table [study|holdout]          per letter: capital and small accuracy, and the mean
  byclass.py one <c> [out.png]              one class (case-sensitive): what it is read as, a sheet of misread
                                            images with the wrong answer, and the terms that cost it / win for the thief
  byclass.py contrast <c> <p> [k]           features separating <c> images read as <p> from <p> images read right
"""
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from charnet.engine import load  # noqa: E402
from eval.mnist import read_idx  # noqa: E402

E = os.path.join(ROOT, "data", "emnist", "gzip")
ARCH, EDITS = os.path.join(ROOT, "arch.json"), os.path.join(ROOT, "weights", "charnet.edits")
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS = CLASSES[10:]
SETS = {"study": (0, 120000, 300), "holdout": (400000, 440000, 120)}
_net = None


def sets():
    path = os.path.join(ROOT, "results", "byclass_sets.npz")
    if not os.path.exists(path):
        X = read_idx("emnist-byclass-train-images-idx3-ubyte.gz", E)
        Y = read_idx("emnist-byclass-train-labels-idx1-ubyte.gz", E)
        out = {}
        for name, (lo, hi, n) in SETS.items():
            idx = np.concatenate([lo + np.nonzero(Y[lo:hi] == c)[0][:n] for c in range(10, 62)])
            out[name + "_X"], out[name + "_y"] = X[idx], np.array([CLASSES[v] for v in Y[idx]])
        np.savez_compressed(path, **out)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def prep(img):
    from gen.handwriting import normalize
    return normalize(img.T.astype(float) / 255.0, "mnist")


def _init():
    global _net
    _net = load(ARCH, EDITS)


def _one(img):
    T = _net.forward(prep(img))
    names = _net.L["L7"]["out"] + _net.L["L8"]["out"]
    return ([T["let:" + c] for c in LETTERS] + [T["cls:" + c] for c in CLASSES], [T[k] for k in names])


def run(name):
    """let:* and cls:* scores and named features for a set, cached until the edit list changes."""
    cache = os.path.join(ROOT, "results", f"byclass_{name}.npz")
    stamp = os.path.getmtime(EDITS)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if float(z["stamp"]) == stamp:
            return z["SL"], z["S"], z["F"], list(z["names"])
    X = sets()[name + "_X"]
    with Pool(8, _init) as p:
        out = p.map(_one, X, chunksize=40)
    sc = np.array([o[0] for o in out])
    F = np.array([o[1] for o in out])
    net = load(ARCH, EDITS)
    names = net.L["L7"]["out"] + net.L["L8"]["out"]
    np.savez(cache, SL=sc[:, :52], S=sc[:, 52:], F=F, names=np.array(names), stamp=stamp)
    return sc[:, :52], sc[:, 52:], F, names


def answers(name):
    SL, S, F, names = run(name)
    y = sets()[name + "_y"]
    a = np.array(list(LETTERS))[SL.argmax(1)]
    return y, a, np.char.upper(a) == np.char.upper(y), SL, S, F, names


def table(name="study", quiet=False):
    y, a, ok, *_ = answers(name)
    acc = {c: ok[y == c].mean() for c in LETTERS}
    if not quiet:
        print(f"{name}: mean over 52 classes {100 * np.mean(list(acc.values())):.1f}%   "
              f"(letters only, case-folded; {len(y)} images)")
        print("  " + "  ".join(f"{U}{100 * acc[U]:3.0f} {U.lower()}{100 * acc[U.lower()]:3.0f}" for U in LETTERS[:13]))
        print("  " + "  ".join(f"{U}{100 * acc[U]:3.0f} {U.lower()}{100 * acc[U.lower()]:3.0f}" for U in LETTERS[13:26]))
    return acc


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        table(sys.argv[2] if len(sys.argv) > 2 else "study")
    elif cmd == "one":
        c = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else None
        y, a, ok, SL, S, F, names = answers("study")
        X = sets()["study_X"]
        mine = y == c
        bad = np.nonzero(mine & ~ok)[0]
        print(f"{c}: {100 * ok[mine].mean():.1f}% of {mine.sum()}   read as: " +
              " ".join(f"{p}:{n}" for p, n in Counter(a[bad]).most_common(10)))
        net = load(ARCH, EDITS)
        Wl, Ll = net.P["L9"]["W"], net.L["L9"]
        for P, _ in Counter(a[bad]).most_common(3):
            pick = bad[a[bad] == P][:40]
            for cls in (c, P):
                styles = net.L["L10"]["groups"]["best:" + cls]
                won, cost = Counter(), Counter()
                for j in pick:
                    T = net.forward(prep(X[j]))
                    st = max(styles, key=lambda k: T[k])
                    won[st] += 1
                    o = Ll["out"].index(st)
                    for i in np.nonzero(Wl[o])[0]:
                        cost[Ll["in"][i]] += Wl[o, i] * T[Ll["in"][i]] / len(pick)
                neg = [f"{k} {v:+.2f}" for k, v in sorted(cost.items(), key=lambda kv: kv[1])[:5] if v < -0.02]
                pos = [f"{k} {v:+.2f}" for k, v in sorted(cost.items(), key=lambda kv: -kv[1])[:6] if v > 0.02]
                print(f"  [{c}->{P}] {cls}: {dict(won.most_common(2))}  costs: {', '.join(neg) or '-'}  |  earns: {', '.join(pos)}")
        if out:
            from PIL import Image, ImageDraw
            n, cell = 24, 56
            rng = np.random.default_rng(5)
            rows = [("ok", rng.permutation(np.nonzero(mine & ok)[0])[:n])] + \
                   [("miss", rng.permutation(bad)[k * n:(k + 1) * n]) for k in range(3)]
            im = Image.new("RGB", ((n + 1) * cell, 4 * (cell + 18)), (40, 40, 40))
            d = ImageDraw.Draw(im)
            for r, (lab, js) in enumerate(rows):
                d.text((6, r * (cell + 18) + 20), lab, fill=(255, 200, 0))
                for k, j in enumerate(js):
                    tile = Image.fromarray((prep(X[j]) * 255).astype(np.uint8)).resize((cell - 4, cell - 4), Image.NEAREST)
                    im.paste(tile, ((k + 1) * cell, r * (cell + 18)))
                    d.text(((k + 1) * cell + 20, r * (cell + 18) + cell - 2), a[j], fill=(120, 200, 255))
            im.save(out)
    elif cmd == "contrast":
        c, p = sys.argv[2], sys.argv[3]
        k = int(sys.argv[4]) if len(sys.argv) > 4 else 14
        y, a, ok, SL, S, F, names = answers("study")
        A = F[(y == c) & (a == p)]
        B = F[(y == p) & ok] if (y == p).any() else F[(np.char.upper(y) == p.upper()) & ok]
        sep = np.abs(A.mean(0) - B.mean(0)) / (np.sqrt((A.var(0) + B.var(0)) / 2) + 0.05)
        o = np.argsort(-sep)[:k]
        print(f"{c} read as {p} ({len(A)}) vs {p} read right ({len(B)}): " +
              "  ".join(f"{names[i]} {A[:, i].mean():.2f}/{B[:, i].mean():.2f}({sep[i]:.1f})" for i in o))
