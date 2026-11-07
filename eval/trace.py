"""Trace one TRAINING image: top classes, key features, and the terms of chosen styles.
Usage: trace.py <train_index> [style ...] [--maps out.png]"""
import os, sys
import numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.study import X, Y, NET  # noqa: E402
from eval.explain import explain  # noqa: E402

KEY = ["loop", "loop_hi", "loop_lo", "loop_two", "loop_tall", "waist", "gap", "narrow", "stem_l", "stem_r",
       "hole@T", "hole@M", "hole@B", "bay_R@T", "bay_R@M", "bay_R@B", "bay_L@T", "bay_L@M", "bay_L@B",
       "bay_U@T", "bay_U@B", "bay_D@T", "bay_D@B", "mid_h@M", "edge_t@T", "edge_b@B",
       "end_any@Tl", "end_any@Tc", "end_any@Tr", "end_any@Ml", "end_any@Mr", "end_any@Bl", "end_any@Bc", "end_any@Br"]
args = sys.argv[1:]
maps_out = None
if "--maps" in args:
    k = args.index("--maps"); maps_out = args[k + 1]; del args[k:k + 2]
i = int(args[0])
T = NET.forward(X[i].astype(float) / 255.0)
styles = NET.L["L9"]["out"]
ranked = sorted(styles, key=lambda s: -T[s])
print(f"train[{i}] label {Y[i]}: " + "  ".join(f"{s[2:]} {T[s]:.2f}" for s in ranked[:5]))
print("  " + " ".join(f"{k}={T[k]:.2f}" for k in KEY if T[k] > 0.05))
for s in args[1:] or [ranked[0][2:]]:
    print("  " + explain(NET, T, "s:" + s, 10))
print("  hole px:", int(T["hole"].sum()), " hole_x px:", int(T["hole_x"].sum()), " ink px:", int(T["ink"].sum()))
if maps_out:
    M = ["x", "ink", "ink_fat", "hole_x", "hole_f", "hole", "bay_R", "bay_L", "bay_U", "bay_D", "edge_t", "edge_b", "mid_h"]
    s = np.concatenate([np.pad(np.clip(T[m], 0, 1), 1, constant_values=.4) for m in M], 1)
    Image.fromarray((s * 255).astype(np.uint8)).resize((s.shape[1] * 4, s.shape[0] * 4), Image.NEAREST).save(maps_out)
