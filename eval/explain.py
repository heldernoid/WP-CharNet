"""Explain a decision: top styles and the terms behind them. Usage: explain.py <char> [style] [seed] [mode]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from charnet.engine import load
from gen.handwriting import sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def explain(net, T, style, k=12):
    L = net.L["L9"]; P = net.P["L9"]
    o = L["out"].index(style)
    terms = [(L["in"][i], P["W"][o, i], T[L["in"][i]]) for i in np.nonzero(P["W"][o])[0]]
    terms.sort(key=lambda t: -abs(t[1] * t[2]))
    return f"{style} = {T[style]:.2f}: " + "  ".join(f"{n}({w:+.2f}x{v:.2f})" for n, w, v in terms[:k])


if __name__ == "__main__":
    ch = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    seeds = range(int(sys.argv[3])) if len(sys.argv) > 3 else range(4)
    net = load(os.path.join(ROOT, "arch.json"), os.path.join(ROOT, "weights", "charnet.edits"))
    styles = net.L["L9"]["out"]
    for seed in seeds:
        mode = "mnist" if seed % 2 else "emnist"
        img, st = sample(ch, style, seed=seed, mode=mode)
        T = net.forward(img)
        ranked = sorted(styles, key=lambda s: -T[s])
        print(f"--- {ch}/{st} seed {seed} {mode}: " + "  ".join(f"{s[2:]} {T[s]:.2f}" for s in ranked[:4]))
        print("   " + explain(net, T, f"s:{ch}/{st}"))
        if ranked[0] != f"s:{ch}/{st}":
            print("   " + explain(net, T, ranked[0]))
