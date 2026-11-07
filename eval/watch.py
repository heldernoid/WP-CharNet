"""Regenerate and print the study-set accuracy (letters only, case-folded) of the classes to watch.
Usage: watch.py <classes, e.g. aAgQ>   (TRAINING data only; see byclass.py)"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eval.byclass as B  # noqa: E402

if __name__ == "__main__":
    print(subprocess.run([sys.executable, os.path.join(ROOT, "gen", "generate.py")], capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()[-1])
    from charnet.engine import load
    load(B.ARCH, B.EDITS)   # fail here, not inside the worker processes
    acc = B.table("study", quiet=True)
    print(f"mean {100 * np.mean(list(acc.values())):.1f}%   " + "  ".join(f"{c}:{100 * acc[c]:.0f}" for c in sys.argv[1] if c in acc))
