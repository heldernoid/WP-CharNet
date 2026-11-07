"""Letters on the EMNIST Letters TRAINING split only. Never point this at a test split.

EMNIST Letters merges each capital with its small letter (label 1 = A or a, ... 26 = Z or z), so the scores here
are case-folded: answering 'a' for an A counts as right. Images get the demo's preparation: transpose (EMNIST
stores them transposed), then normalize(img, "mnist").

Chunks of the training split:
  study   train[0:10400]       decisions are made here (images may be viewed)
  holdout train[60000:65200]   never viewed; totals checked once per round

Usage:
  letters.py score [study|holdout]          accuracy (letters-only and all-62, case-folded), per letter, top confusions
  letters.py sheet <out.png> <L> [n]        random study images of letter L
  letters.py errors <out.png> <L> <P> [n]   study images of L read as P (P: a letter, any case, or a digit)
  letters.py contrast <L> <P> [k] [--err]   features that separate L from P (--err: L read as P vs P read right)
  letters.py terms <L> <P> [n]              for study images of L read as P (letters-only): each class's winning
                                            style and the average part every term played in it
  letters.py cross <d> <L> [k] [--err]      MNIST digit d (train[0:6000]) vs EMNIST letter L (study chunk);
                                            --err: digit images read as L vs letter images read as L
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from charnet.engine import load  # noqa: E402
from eval.mnist import read_idx  # noqa: E402
from eval.study import save_sheet  # noqa: E402

E = os.path.join(ROOT, "data", "emnist", "gzip")
ARCH, EDITS = os.path.join(ROOT, "arch.json"), os.path.join(ROOT, "weights", "charnet.edits")
CHUNKS = {"study": range(0, 10400), "holdout": range(60000, 65200)}
LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_net = None


def data():
    X = read_idx("emnist-letters-train-images-idx3-ubyte.gz", E)
    Y = read_idx("emnist-letters-train-labels-idx1-ubyte.gz", E)
    return X, np.array([LET[v - 1] for v in Y])


def prep(img):
    from gen.handwriting import normalize
    return normalize(img.T.astype(float) / 255.0, "mnist")


def _init():
    global _net
    _net = load(ARCH, EDITS)


def _one(img):
    T = _net.forward(prep(img))
    names = _net.L["L7"]["out"] + _net.L["L8"]["out"]
    return [T["cls:" + c] for c in _net.arch["classes"]], [T[k] for k in names], [T["let:" + c] for c in LETTERS62]


LETTERS62 = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]


def let_scores(chunk):
    """Letters-only scores (let:*, letter referees only) for the chunk, from the same cache as run()."""
    run(chunk)
    return np.load(os.path.join(ROOT, "results", f"letters_{chunk}.npz"), allow_pickle=True)["SL"]


def run(chunk):
    """Class scores and named features for a chunk, cached until the edit list changes."""
    cache = os.path.join(ROOT, "results", f"letters_{chunk}.npz")
    stamp = os.path.getmtime(EDITS)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if float(z["stamp"]) == stamp and "SL" in z:
            return z["S"], z["F"], list(z["classes"]), list(z["names"])
    X, _ = data()
    with Pool(8, _init) as p:
        out = p.map(_one, X[CHUNKS[chunk]], chunksize=50)
    net = load(ARCH, EDITS)
    S, F, SL = (np.array([o[k] for o in out]) for k in range(3))
    classes, names = net.arch["classes"], net.L["L7"]["out"] + net.L["L8"]["out"]
    np.savez(cache, S=S, F=F, SL=SL, classes=np.array(classes), names=np.array(names), stamp=stamp)
    return S, F, classes, names


def _digit_one(img):
    T = _net.forward(img.astype(float) / 255.0)
    names = _net.L["L7"]["out"] + _net.L["L8"]["out"]
    return [T["cls:" + c] for c in _net.arch["classes"]], [T[k] for k in names]


def run_digits():
    """The same for MNIST train[0:6000] (the studied digit chunk)."""
    cache = os.path.join(ROOT, "results", "digits_study.npz")
    stamp = os.path.getmtime(EDITS)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if float(z["stamp"]) == stamp:
            return z["S"], z["F"]
    X = read_idx("train-images-idx3-ubyte.gz")[:6000]
    with Pool(8, _init) as p:
        out = p.map(_digit_one, X, chunksize=50)
    S, F = np.array([o[0] for o in out]), np.array([o[1] for o in out])
    np.savez(cache, S=S, F=F, stamp=stamp)
    return S, F


def answers(S, classes, SL=None):
    """Case-folded answer over all 62 classes (cls:*), and over the 52 letters only (let:* when given)."""
    up = np.array([c.upper() for c in classes])
    let = np.array([c.isalpha() for c in classes])
    a62 = up[S.argmax(1)]
    a52 = np.char.upper(np.array(LETTERS62)[SL.argmax(1)]) if SL is not None else up[np.where(let, S, -np.inf).argmax(1)]
    return a62, a52


if __name__ == "__main__":
    cmd = sys.argv[1]
    _, Yall = data()
    if cmd == "score":
        chunk = sys.argv[2] if len(sys.argv) > 2 else "study"
        S, F, classes, names = run(chunk)
        y = Yall[CHUNKS[chunk]]
        a62, a52 = answers(S, classes, let_scores(chunk))
        print(f"{chunk}: letters-only {100 * (a52 == y).mean():.1f}%   all 62 {100 * (a62 == y).mean():.1f}%   "
              f"(answered a digit {100 * np.char.isdigit(a62).mean():.1f}%)")
        print("per letter (letters-only):", " ".join(f"{L}:{100 * (a52[y == L] == L).mean():.0f}" for L in LET))
        if chunk == "study":
            from collections import Counter
            c = Counter((t, p) for t, p in zip(y, a62) if t != p)
            print("top confusions (all 62):", "  ".join(f"{t}->{p} {n}" for (t, p), n in c.most_common(30)))
    elif cmd in ("sheet", "errors"):
        X, _ = data()
        idx = np.array(CHUNKS["study"])
        rng = np.random.default_rng(0)
        if cmd == "sheet":
            out, L = sys.argv[2], sys.argv[3].upper()
            n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
            pick = rng.permutation(idx[Yall[idx] == L])[:n]
        else:
            out, L, P = sys.argv[2], sys.argv[3].upper(), sys.argv[4].upper()
            n = int(sys.argv[5]) if len(sys.argv) > 5 else 40
            S, _, classes, _ = run("study")
            a62, _ = answers(S, classes)
            pick = rng.permutation(idx[(Yall[idx] == L) & (a62 == P)])[:n]
        print(len(pick), "images")
        save_sheet([(prep(X[i]) * 255).astype(np.uint8) for i in pick], out)
    elif cmd == "contrast":
        args = [a for a in sys.argv[2:] if a != "--err"]
        L, P = args[0].upper(), args[1].upper()
        k = int(args[2]) if len(args) > 2 else 14
        S, F, classes, names = run("study")
        y = Yall[CHUNKS["study"]]
        a62, _ = answers(S, classes)
        ma, mb = ((y == L) & (a62 == P), (y == P) & (a62 == P)) if "--err" in sys.argv else (y == L, y == P)
        if not mb.any():   # P is a digit: compare against nothing better than all other letters
            mb = (y != L)
        A, B = F[ma].mean(0), F[mb].mean(0)
        sep = np.abs(A - B) / (np.sqrt((F[ma].var(0) + F[mb].var(0)) / 2) + 0.05)
        order = np.argsort(-sep)[:k]
        print(f"{L} ({ma.sum()}) vs {P} ({mb.sum()}): " + "  ".join(f"{names[i]} {A[i]:.2f}/{B[i]:.2f}({sep[i]:.1f})" for i in order))
    elif cmd == "terms":
        from collections import Counter
        L, P = sys.argv[2].upper(), sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
        S, _, classes, _ = run("study")
        X, _ = data()
        idx = np.array(CHUNKS["study"])
        let = np.array([c.isalpha() for c in classes])
        raw = np.array(classes)[np.where(let, S, -np.inf).argmax(1)]
        pick = idx[(Yall[idx] == L) & (raw == P)][:n]
        net = load(ARCH, EDITS)
        Wl, Ll = net.P["L9"]["W"], net.L["L9"]
        for cls in ([c for c in classes if c.upper() == L] + [P]):
            styles = net.L["L10"]["groups"]["best:" + cls]
            won, contrib = Counter(), {}
            for i in pick:
                T = net.forward(prep(X[i]))
                st = max(styles, key=lambda k: T[k])
                won[st] += 1
                o = Ll["out"].index(st)
                for j in np.nonzero(Wl[o])[0]:
                    contrib.setdefault(st, Counter())[Ll["in"][j]] += Wl[o, j] * T[Ll["in"][j]] / len(pick)
            for st, c in won.most_common(2):
                top = sorted(contrib[st].items(), key=lambda kv: kv[1])
                neg = [f"{k} {v:+.2f}" for k, v in top[:6] if v < -0.01]
                pos = [f"{k} {v:+.2f}" for k, v in top[::-1][:6] if v > 0.01]
                print(f"  {st} (won {c}/{len(pick)}):  hurt by {', '.join(neg)}  |  helped by {', '.join(pos)}")
    elif cmd == "cross":
        args = [a for a in sys.argv[2:] if a != "--err"]
        d, L = args[0], args[1].upper()
        k = int(args[2]) if len(args) > 2 else 14
        S, F, classes, names = run("study")
        Sd, Fd = run_digits()
        yd = read_idx("train-labels-idx1-ubyte.gz")[:6000].astype(str)
        y = Yall[CHUNKS["study"]]
        a62, _ = answers(S, classes)
        ad, _ = answers(Sd, classes)
        if "--err" in sys.argv:
            ma, mb = (yd == d) & (ad == L), (y == L) & (a62 == L)
        else:
            ma, mb = yd == d, y == L
        A, B = Fd[ma].mean(0), F[mb].mean(0)
        sep = np.abs(A - B) / (np.sqrt((Fd[ma].var(0) + F[mb].var(0)) / 2) + 0.05)
        order = np.argsort(-sep)[:k]
        print(f"digit {d} ({ma.sum()}) vs {L} ({mb.sum()}): " + "  ".join(f"{names[i]} {A[i]:.2f}/{B[i]:.2f}({sep[i]:.1f})" for i in order))
