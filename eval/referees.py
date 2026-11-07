"""MNIST TRAINING split only: for misread images, what did the referee of the (true, predicted) pair do?
Usage: referees.py [n] [start]"""
import os, sys
from collections import defaultdict
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.study import X, Y, NET  # noqa: E402

pairs = {tuple(k[5:].split("/")) for k in NET.L["L12"]["out"]}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
stats = defaultdict(list)
for i in range(start, start + n):
    T = NET.forward(X[i] / 255.0)
    s = [T[f"cls:{d}"] for d in range(10)]
    p, t = str(int(np.argmax(s))), str(Y[i])
    if p == t:
        continue
    key = (t, p)
    if (t, p) in pairs:
        a, b = t, p
    elif (p, t) in pairs:
        a, b = p, t
    else:
        stats[key].append(None)
        continue
    gate = T[f"gate:{a}/{b}"]
    pro_true = T[f"pro:{t}>{p}"]
    stats[key].append((T[f"best:{p}"] - T[f"best:{t}"], gate, pro_true, T[f"pro:{p}>{t}"]))
print("true->pred  n  | style gap  gate  pro(true)  pro(pred)   [no referee: -]")
for key, v in sorted(stats.items(), key=lambda kv: -len(kv[1])):
    if len(v) < 5:
        continue
    if v[0] is None:
        print(f"{key[0]}->{key[1]}  {len(v):3d}  | -")
        continue
    a = np.array(v)
    print(f"{key[0]}->{key[1]}  {len(v):3d}  | {a[:,0].mean():.2f}  {a[:,1].mean():.2f}  {a[:,2].mean():.2f}  {a[:,3].mean():.2f}"
          f"   (pro(true)>0.5: {np.mean(a[:,2] > .5):.0%}, gate<0.3: {np.mean(a[:,1] < .3):.0%})")
