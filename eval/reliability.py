"""How often an answer shown at a given percentage was right, on UNSTUDIED training chunks only
(MNIST train[6000:9000], EMNIST Letters train[60000:65200]). Writes web/model/reliability.json for the demo.
The temperature is not changed here; this only measures it.
Usage: reliability.py [arch.json] [edits]   (default: the current network)"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eval.letters as LT  # noqa: E402
from charnet.engine import load  # noqa: E402
from eval.mnist import read_idx  # noqa: E402

TEMPERATURE = 0.5   # as typed in web/charnet.js
EDGES = [0, .1, .2, .3, .5, .7, .9, .97, 1.0001]
_net = None


def _init(a, e):
    global _net
    _net = load(a, e)


def _scores(job):
    img, emnist = job
    x = LT.prep(img) if emnist else img.astype(float) / 255.0
    T = _net.forward(x)
    C = _net.arch["classes"]
    # "digits only" / "letters only" read the dig:* / let:* outputs (only referees within that kind)
    return ([T["cls:" + c] for c in C] + [T.get("dig:" + c, T["cls:" + c]) for c in C[:10]]
            + [T.get("let:" + c, T["cls:" + c]) for c in C[10:]])


def bins(S, right):
    z = np.exp((S - S.max(1, keepdims=True)) / TEMPERATURE)
    conf = (z / z.sum(1, keepdims=True)).max(1)
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 20:
            out.append({"lo": lo, "hi": min(hi, 1), "n": int(m.sum()), "shown": round(float(conf[m].mean()), 3),
                        "right": round(float(right[m].mean()), 3)})
    return out


if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "arch.json")
    edits = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "weights", "charnet.edits")
    Xd = read_idx("train-images-idx3-ubyte.gz")[6000:9000]
    yd = read_idx("train-labels-idx1-ubyte.gz")[6000:9000].astype(str)
    Xl, Yl = LT.data()
    Xl, yl = Xl[LT.CHUNKS["holdout"]], Yl[LT.CHUNKS["holdout"]]
    with Pool(8, _init, (arch, edits)) as p:
        Sd = np.array(p.map(_scores, [(x, False) for x in Xd], chunksize=50))
        Sl = np.array(p.map(_scores, [(x, True) for x in Xl], chunksize=50))
    C = np.array(load(arch, edits).arch["classes"])
    up = np.char.upper
    Dd, Ll = Sd[:, 62:72], Sl[:, 72:]          # dig:* on digit images, let:* on letter images
    Sd, Sl = Sd[:, :62], Sl[:, :62]
    res = {
        "temperature": TEMPERATURE,
        "digits": {"digits": bins(Dd, C[:10][Dd.argmax(1)] == yd)},
        "all": {"digits": bins(Sd, C[Sd.argmax(1)] == yd), "letters": bins(Sl, up(C[Sl.argmax(1)]) == yl)},
        "letters": {"letters": bins(Ll, up(C[10:][Ll.argmax(1)]) == yl)},
        "accuracy": {"digits_only": float((C[:10][Dd.argmax(1)] == yd).mean()),
                     "digits_all62": float((C[Sd.argmax(1)] == yd).mean()),
                     "letters_only": float((up(C[10:][Ll.argmax(1)]) == yl).mean()),
                     "letters_all62": float((up(C[Sl.argmax(1)]) == yl).mean())},
    }
    out = os.path.join(ROOT, "web", "model", "reliability.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["accuracy"]), "->", out)
    for mode, d in res.items():
        if isinstance(d, dict) and mode not in ("accuracy",):
            for k, b in d.items():
                print(mode, k, " ".join(f"[{x['lo']}-{x['hi']:.2f} shown {x['shown']:.2f} right {x['right']:.2f} n={x['n']}]" for x in b))
