"""Test WP-CharNet on the MNIST or EMNIST Digits test split. Writes aggregate results only (no images).

Usage: python eval/mnist.py <arch.json> <edits> <out.json> [--emnist]
  --emnist: EMNIST Digits test (data/emnist/gzip/emnist-digits-test-*.gz). Each image is transposed (EMNIST
            stores them transposed) and normalised like a drawn character: box into 20x20, centre of mass centred.
"""
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from charnet.engine import load  # noqa: E402

DATA = os.path.join(ROOT, "data", "mnist")
DIGITS = list("0123456789")
_net = None
_emnist = False


def read_idx(name, data=DATA):
    with gzip.open(os.path.join(data, name)) as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _init(arch, edits, emnist=False):
    global _net, _emnist
    _net = load(arch, edits)
    _emnist = emnist


def _emnist_prep(img):
    from gen.handwriting import normalize
    return normalize(img.T.astype(float) / 255.0, "mnist")


def _classify(img):
    x = _emnist_prep(img) if _emnist else img.astype(float) / 255.0
    T = _net.forward(x)
    classes = _net.arch["classes"]
    s = np.array([T["cls:" + c] for c in classes])
    full = classes[int(np.argmax(s))]
    digit = DIGITS[int(np.argmax(s[:10]))]
    return full, digit


def main(arch, edits, out, emnist=False):
    if emnist:
        E = os.path.join(ROOT, "data", "emnist", "gzip")
        X = read_idx("emnist-digits-test-images-idx3-ubyte.gz", E)
        y = read_idx("emnist-digits-test-labels-idx1-ubyte.gz", E)
    else:
        X = read_idx("t10k-images-idx3-ubyte.gz")
        y = read_idx("t10k-labels-idx1-ubyte.gz")
    t = time.time()
    with Pool(os.cpu_count(), initializer=_init, initargs=(arch, edits, emnist)) as p:
        preds = p.map(_classify, list(X), chunksize=64)
    full = [a for a, _ in preds]
    dig = [b for _, b in preds]
    truth = [str(v) for v in y]
    conf = [[0] * 10 for _ in range(10)]
    for tr, pr in zip(truth, dig):
        conf[int(tr)][int(pr)] += 1
    res = {
        "split": f"EMNIST Digits test ({len(y):,})" if emnist else "MNIST test (10,000)",
        "edits_file": os.path.basename(edits),
        "digits_only_accuracy": float(np.mean([a == b for a, b in zip(truth, dig)])),
        "strict_62class_accuracy": float(np.mean([a == b for a, b in zip(truth, full)])),
        "per_digit": {d: {"n": sum(conf[i]), "correct": conf[i][i]} for i, d in enumerate(DIGITS)},
        "confusion_digits_only": conf,
        "non_digit_answers": {c: full.count(c) for c in sorted(set(full)) if c not in DIGITS},
        "seconds": round(time.time() - t, 1),
    }
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion_digits_only", "non_digit_answers")}, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--emnist"]
    main(*args, emnist="--emnist" in sys.argv)
