"""Generate WP-CharNet: arch.json (every map, feature and neuron named) and the edit list.

No weight here comes from data. Every value is written below by hand, with a reason.
Run:  python gen/generate.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

lines = []


def emit(s, why=None):
    lines.append(s if why is None else f"{s}   # {why}")


def section(title):
    lines.append("")
    lines.append(f"# ---- {title} " + "-" * max(0, 70 - len(title)))


def fmt(v):
    return f"{v:g}"


# =============================================================================
# L1  ink: binarise the grey input
# =============================================================================
L1 = {"id": "L1", "op": "conv", "in": ["x"], "out": ["ink", "ink_core"], "k": 1, "act": "clip01"}


def gen_L1():
    section("L1 ink: 0 below grey 0.25, 1 above grey 0.5")
    emit("TAP L1 ink x 0 0 4")
    emit("BIAS L1 ink -1", "ink = clip(4x - 1): anti-aliased edges count as half ink")
    emit("TAP L1 ink_core x 0 0 5")
    emit("BIAS L1 ink_core -3.5", "ink_core = only near-white pixels (>= 0.9): the grey inside a tight loop is not a wall")


# =============================================================================
# L2  strokes and stroke ends (9x9 kernels on ink)
# =============================================================================
# Directions in (dy, dx), y pointing down.
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
        "ul": (-1, -1), "ur": (-1, 1), "dl": (1, -1), "dr": (1, 1)}
# Runs are read from ink_fat (1 px thicker), so a slanted stroke still has a straight core,
# and are long enough (9 px each way) that a thick or slanted stroke of one direction cannot
# contain a run of another: a stem leaning 20 degrees is still vertical, not diagonal.
RUNS = {"run_h": [(0, d) for d in range(-4, 5)],                 # 9 px horizontal
        "run_v": [(d, 0) for d in range(-4, 5)],                 # 9 px vertical
        "run_d1": [(-d, d) for d in range(-4, 5)],               # 9 px '/' (13 px long)
        "run_d2": [(d, d) for d in range(-4, 5)]}                # 9 px '\'
ENDS = [f"end_{d}" for d in DIRS]
L2 = {"id": "L2", "op": "conv", "in": ["ink", "ink_fat"], "out": list(RUNS) + ENDS, "k": 9, "act": "clip01"}


def end_taps(d):
    """Taps for 'a stroke ends here, pointing in direction d'.

    body : ink on the line going back from the tip (the stroke itself)
    front: no ink in a wedge ahead of the tip (the stroke really stops)
    sides: no ink 4 px to either side (rules out the flat edge of a thick bar)
    shoulders: no ink beside the stroke 4 px back (rules out a sharp corner, where a
               second stroke leaves the tip at an angle, as at the top right of a 7)
    """
    dy, dx = d
    n = math.hypot(dy, dx)
    uy, ux = dy / n, dx / n          # unit along d
    py, px = ux, -uy                 # unit perpendicular
    body, front, sides, shoulders = [], [], [], []
    for oy in range(-4, 5):
        for ox in range(-4, 5):
            along = oy * uy + ox * ux
            across = oy * py + ox * px
            if abs(across) < 0.3 and -4.6 <= along <= 0.01:
                body.append((oy, ox))
            elif 0.9 <= along <= 3.3 and abs(across) <= 2.3:
                front.append((oy, ox))
            elif abs(along) <= 0.8 and 3.5 <= abs(across) <= 4.6:
                sides.append((oy, ox))
            elif -5.0 <= along <= -2.5 and 3.2 <= abs(across) <= 4.8:
                shoulders.append((oy, ox))
    return body, front, sides, shoulders


def gen_L2():
    section("L2 strokes: a run detector fires only if every pixel of its line is ink (AND via bias)")
    for name, taps in RUNS.items():
        for dy, dx in taps:
            emit(f"TAP L2 {name} ink_fat {dy} {dx} 1")
        emit(f"BIAS L2 {name} {-(len(taps) - 1)}", f"{name}: {len(taps)} pixels of thickened ink in a row")
    section("L2 stroke ends: stroke behind, nothing ahead, nothing beside")
    for d, v in DIRS.items():
        body, front, sides, shoulders = end_taps(v)
        for dy, dx in body:
            emit(f"TAP L2 end_{d} ink {dy} {dx} 1")
        for dy, dx in front:
            emit(f"TAP L2 end_{d} ink {dy} {dx} -2")
        for dy, dx in sides:
            emit(f"TAP L2 end_{d} ink {dy} {dx} -2")
        for dy, dx in shoulders:
            emit(f"TAP L2 end_{d} ink {dy} {dx} -1")
        emit(f"BIAS L2 end_{d} {-(len(body) - 1)}",
             f"end_{d}: {len(body)} px of stroke behind; empty: {len(front)} px ahead, "
             f"{len(sides)} beside, {len(shoulders)} at the shoulders")


# =============================================================================
# L3  rays: is there ink anywhere to the left / right / above / below?
# =============================================================================
L3 = {"id": "L3", "op": "ray", "in": "ink",
      "out": {"inkL": "left", "inkR": "right", "inkU": "up", "inkD": "down"}}

# =============================================================================
# LX  scan lines: how many strokes does each row / column cross? and the side profiles
# =============================================================================
# A stroke starts on a row where ink rises from the pixel on its left; summing the rises over
# the row counts the strokes the row crosses (an 8 reads 2,1,2 top to bottom, a 0 reads 1,2,1).
LX1 = {"id": "LX1", "op": "conv", "in": ["ink", "inkL", "inkR", "inkU", "inkD"],
       "out": ["start_h", "start_v", "marg_l", "marg_r", "marg_t", "marg_b"], "k": 3, "act": "clip01"}
LX2 = {"id": "LX2", "op": "linesum", "dir": "row", "in": "start_h", "out": "cnt_h"}
LX3 = {"id": "LX3", "op": "linesum", "dir": "col", "in": "start_v", "out": "cnt_v"}
GE = [1, 2, 3]
LX4 = {"id": "LX4", "op": "conv", "in": ["cnt_h", "cnt_v"],
       "out": [f"{a}_ge{k}" for a in "hv" for k in GE], "k": 1, "act": "clip01"}
CROSS_MAPS = ["hx1", "hx2", "hx3", "vx1", "vx2", "vx3"]
LX5 = {"id": "LX5", "op": "conv", "in": LX4["out"], "out": CROSS_MAPS, "k": 1, "act": "clip01"}

# LD  the same count along the two diagonals: how many strokes does each / line and each \ line cross?
# A V's / lines cross one arm then the other; an X's centre lines cross both strokes; a K's \ lines
# through its stem cross stem and arm. Diagonal lines see slanted strokes edge-on, as rows see a stem.
LD1 = {"id": "LD1", "op": "conv", "in": ["ink"], "out": ["start_d1", "start_d2"], "k": 3, "act": "clip01"}
LD2 = {"id": "LD2", "op": "linesum", "dir": "d1", "in": "start_d1", "out": "cnt_d1"}
LD3 = {"id": "LD3", "op": "linesum", "dir": "d2", "in": "start_d2", "out": "cnt_d2"}
LD4 = {"id": "LD4", "op": "conv", "in": ["cnt_d1", "cnt_d2"],
       "out": [f"{a}_ge{k}" for a in ("d1", "d2") for k in GE], "k": 1, "act": "clip01"}
DIAG_MAPS = ["ax1", "ax2", "ax3", "bx1", "bx2", "bx3"]   # a: along / lines, b: along \ lines
LD5 = {"id": "LD5", "op": "conv", "in": LD4["out"], "out": DIAG_MAPS, "k": 1, "act": "clip01"}


def gen_LD():
    section("LD1 stroke starts along / lines and \\ lines")
    emit("TAP LD1 start_d1 ink 0 0 1")
    emit("TAP LD1 start_d1 ink -1 1 -1", "ink rises from the pixel up-right: a stroke starts here on its / line")
    emit("TAP LD1 start_d2 ink 0 0 1")
    emit("TAP LD1 start_d2 ink -1 -1 -1", "ink rises from the pixel up-left: a stroke starts here on its \\ line")
    section("LD4 diagonal crossing count >= k (same soft step as the rows)")
    for a in ("d1", "d2"):
        for k in GE:
            emit(f"TAP LD4 {a}_ge{k} cnt_{a} 0 0 2")
            emit(f"BIAS LD4 {a}_ge{k} {fmt(-2 * k + 1.5)}", f"the {'/' if a == 'd1' else 'backslash'} line crosses at least {k} strokes")
    section("LD5 diagonal crossing count exactly 1, exactly 2, 3 or more")
    for a, n in (("d1", "a"), ("d2", "b")):
        emit(f"TAP LD5 {n}x1 {a}_ge1 0 0 1")
        emit(f"TAP LD5 {n}x1 {a}_ge2 0 0 -1", f"{n}x1: exactly one stroke on this diagonal")
        emit(f"TAP LD5 {n}x2 {a}_ge2 0 0 1")
        emit(f"TAP LD5 {n}x2 {a}_ge3 0 0 -1", f"{n}x2: exactly two strokes")
        emit(f"TAP LD5 {n}x3 {a}_ge3 0 0 1", f"{n}x3: three or more strokes")


def gen_LX():
    section("LX1 stroke starts along rows / columns, and side margins")
    emit("TAP LX1 start_h ink 0 0 1")
    emit("TAP LX1 start_h ink 0 -1 -1", "ink rises from the left neighbour: a stroke starts here (row scan)")
    emit("TAP LX1 start_v ink 0 0 1")
    emit("TAP LX1 start_v ink -1 0 -1", "ink rises from the pixel above: a stroke starts here (column scan)")
    for name, near, far, why in [("marg_l", "inkR", "inkL", "background left of the character on its row (left profile)"),
                                 ("marg_r", "inkL", "inkR", "background right of the character (right profile)"),
                                 ("marg_t", "inkD", "inkU", "background above the character in its column"),
                                 ("marg_b", "inkU", "inkD", "background below the character")]:
        emit(f"TAP LX1 {name} {near} 0 0 1")
        emit(f"TAP LX1 {name} {far} 0 0 -1")
        emit(f"TAP LX1 {name} ink 0 0 -1", why)
    section("LX4 crossing count >= k (soft step between k-0.75 and k-0.25: half-grey edges are not a stroke)")
    for a in "hv":
        for k in GE:
            emit(f"TAP LX4 {a}_ge{k} cnt_{a} 0 0 2")
            emit(f"BIAS LX4 {a}_ge{k} {fmt(-2 * k + 1.5)}", f"{'row' if a == 'h' else 'column'} crosses at least {k} strokes")
    section("LX5 crossing count exactly 1, exactly 2, 3 or more")
    for a in "hv":
        emit(f"TAP LX5 {a}x1 {a}_ge1 0 0 1")
        emit(f"TAP LX5 {a}x1 {a}_ge2 0 0 -1", f"{a}x1: exactly one stroke on this line")
        emit(f"TAP LX5 {a}x2 {a}_ge2 0 0 1")
        emit(f"TAP LX5 {a}x2 {a}_ge3 0 0 -1", f"{a}x2: exactly two strokes (the two sides of a loop or a U)")
        emit(f"TAP LX5 {a}x3 {a}_ge3 0 0 1", f"{a}x3: three or more strokes (E, 3 and 5 read down a column)")


# =============================================================================
# LJ  branches: how many strokes leave this point? (1 = an end, 2 = a stroke, 3 = a T or Y, 4 = a crossing)
# =============================================================================
# 16 points on a circle of radius 5 around the pixel. Going round the circle, every step from
# background onto ink is one stroke crossing the circle; the sum counts the strokes.
RING = [(round(-5 * math.sin(math.radians(22.5 * k))), round(5 * math.cos(math.radians(22.5 * k)))) for k in range(16)]
LJ0 = {"id": "LJ0", "op": "conv", "in": ["ink"], "out": ["ink_ring"], "k": 1, "act": "clip01"}
LJ1 = {"id": "LJ1", "op": "conv", "in": ["ink_ring"], "out": [f"arc{k}" for k in range(16)], "k": 11, "act": "clip01"}
LJ2 = {"id": "LJ2", "op": "conv", "in": ["ink"] + LJ1["out"], "out": ["branches"], "k": 1, "act": "relu"}
LJ3 = {"id": "LJ3", "op": "conv", "in": ["branches", "ink_core"], "out": ["junction", "crossing"], "k": 1, "act": "clip01"}


def gen_LJ():
    section("LJ0 ring ink: the grey fringe of a stroke counts as background")
    emit("TAP LJ0 ink_ring ink 0 0 2")
    emit("BIAS LJ0 ink_ring -1", "sharp ink: two arms leaving a crossing at a shallow angle stay apart "
         "(on thickened ink they merge and an 8's crossing reads as a plain stroke)")
    section("LJ1 ring steps: ink at ring point k but not at point k-1 (16 points, radius 5)")
    for k, (dy, dx) in enumerate(RING):
        py, px = RING[k - 1]
        emit(f"TAP LJ1 arc{k} ink_ring {dy} {dx} 1")
        emit(f"TAP LJ1 arc{k} ink_ring {py} {px} -1", f"arc{k}: a stroke crosses the ring between {k - 1} and {k}")
    section("LJ2 branches = number of strokes crossing the ring, counted only on ink")
    for k in range(16):
        emit(f"TAP LJ2 branches arc{k} 0 0 1")
    emit("TAP LJ2 branches ink 0 0 4")
    emit("BIAS LJ2 branches -4", "off the stroke the count is pushed below zero (relu)")
    section("LJ3 junction (3+ strokes meet) and crossing (4+), centred on the solid core of the stroke")
    emit("TAP LJ3 junction branches 0 0 1")
    emit("TAP LJ3 junction ink_core 0 0 2")
    emit("BIAS LJ3 junction -4.5", "junction: 3 strokes leave this point (T, Y, the join of a 9's tail)")
    emit("TAP LJ3 crossing branches 0 0 1")
    emit("TAP LJ3 crossing ink_core 0 0 2")
    emit("BIAS LJ3 crossing -5.5", "crossing: 4 strokes leave this point (the X in the middle of an 8, the cross of a 4)")


# =============================================================================
# L4-L6  outside, holes, bays
# =============================================================================
L3F = {"id": "L3F", "op": "max3", "in": "ink", "out": "ink_fat"}
# Relational strokes: where a stroke sits relative to the rest of the character, not the frame.
# Stroke edges, and rays over them: "is there a DIFFERENT stroke to the left/right/above/below?"
# A ray over plain ink would see the thick body of the stroke itself; a ray over the far edge
# of strokes only sees a stroke that is separated from this one by background.
EDGES = {  # name: (dy, dx) of the neighbour that must be background
    "eR": (0, 1), "eL": (0, -1), "eB": (1, 0), "eT": (-1, 0)}
L3S = {"id": "L3S", "op": "conv", "in": ["ink"], "out": list(EDGES), "k": 3, "act": "clip01"}
L3T = {"id": "L3T", "op": "ray", "in": "eR", "out": {"sepL": "left"}}
L3U = {"id": "L3U", "op": "ray", "in": "eL", "out": {"sepR": "right"}}
L3V = {"id": "L3V", "op": "ray", "in": "eB", "out": {"sepU": "up"}}
L3W = {"id": "L3W", "op": "ray", "in": "eT", "out": {"sepD": "down"}}
# Relational strokes: where a stroke sits relative to the rest of the character, not the frame.
REL = {  # name: (run, [(ray, dy, dx, weight)], bias, why)
    "edge_l": ("run_v", [("sepL", 0, 0, -2)], 0, "vertical stroke with no other stroke to its left (left stem or side)"),
    "edge_r": ("run_v", [("sepR", 0, 0, -2)], 0, "vertical stroke with no other stroke to its right"),
    "edge_t": ("run_h", [("sepU", 0, 0, -2)], 0, "horizontal stroke with no other stroke above it (top bar, top of a curve)"),
    "edge_b": ("run_h", [("sepD", 0, 0, -2)], 0, "horizontal stroke with no other stroke below it"),
    "mid_h": ("run_h", [("sepU", 0, 0, 1), ("sepD", 0, 0, 1)], -2, "horizontal stroke with other strokes above and below (E, H, A, 4)"),
    "mid_v": ("run_v", [("sepL", 0, 0, 1), ("sepR", 0, 0, 1)], -2, "vertical stroke with other strokes on both sides (m, 4, t, Y)"),
}
L3R = {"id": "L3R", "op": "conv", "in": ["run_h", "run_v", "sepL", "sepR", "sepU", "sepD"], "out": list(REL),
       "k": 1, "act": "clip01"}
L3G = {"id": "L3G", "op": "ray", "in": "ink_fat",
       "out": {"fatL": "left", "fatR": "right", "fatU": "up", "fatD": "down"}}
L3H = {"id": "L3H", "op": "ray", "in": "ink_core",
       "out": {"coreL": "left", "coreR": "right", "coreU": "up", "coreD": "down"}}
L4 = {"id": "L4", "op": "conv", "in": ["inkL", "inkR", "inkU", "inkD", "ink", "fatL", "fatR", "fatU", "fatD", "ink_fat",
                                      "coreL", "coreR", "coreU", "coreD", "ink_core"],
      "out": ["outside", "outside_f", "outside_c"], "k": 1, "act": "clip01"}
L5 = {"id": "L5", "op": "loop", "n": 24, "body": [
    {"id": "L5a", "op": "plusmax", "in": "outside", "out": "outside_nb"},
    {"id": "L5c", "op": "plusmax", "in": "outside_f", "out": "outside_f_nb"},
    {"id": "L5d", "op": "plusmax", "in": "outside_c", "out": "outside_c_nb"},
    {"id": "L5b", "op": "conv", "in": ["outside_nb", "ink", "outside_f_nb", "ink_fat", "outside_c_nb", "ink_core"],
     "out": ["outside", "outside_f", "outside_c"], "k": 1, "act": "clip01"}]}
BAYS = {"bay_R": ("inkL", "inkU", "inkD", "inkR"),   # open to the right: walls left, above, below
        "bay_L": ("inkR", "inkU", "inkD", "inkL"),
        "bay_U": ("inkL", "inkR", "inkD", "inkU"),
        "bay_D": ("inkL", "inkR", "inkU", "inkD")}
L6 = {"id": "L6", "op": "conv", "in": ["outside", "ink", "inkL", "inkR", "inkU", "inkD", "outside_f", "ink_fat",
                                      "outside_c", "ink_core"],
      "out": ["hole_x", "hole_f", "hole_c"] + list(BAYS) + ["gap_v"], "k": 1, "act": "clip01"}
L6B = {"id": "L6B", "op": "conv", "in": ["hole_x", "hole_f", "hole_c"], "out": ["hole"], "k": 1, "act": "clip01"}
L6C = {"id": "L6C", "op": "ray", "in": "hole", "out": {"holeU": "up", "holeD": "down"}}
L6D = {"id": "L6D", "op": "conv", "in": ["ink", "holeU", "holeD"], "out": ["between_holes"], "k": 1, "act": "clip01"}
# Stems beside a bowl, wherever the letter sits: a narrow b is centred in the frame, so "stem in the left third"
# misses it; "a vertical stroke with a hole to its right" does not. (b, p, B, D, P, R / d, q, a, 9)
L6E = {"id": "L6E", "op": "ray", "in": "hole", "out": {"holeL": "left", "holeR": "right"}}
L6F = {"id": "L6F", "op": "conv", "in": ["run_v", "holeL", "holeR"], "out": ["stem_lh", "stem_rh"], "k": 1, "act": "clip01"}
# A bowl's own sides are short strokes beside its hole too; what makes b/d/p/q is the stem CONTINUING past the bowl:
# ascender (b, d) above it, descender (p, q) below it, on rows where there is no hole at all.
L6G = {"id": "L6G", "op": "ray", "in": "stem_lh", "out": {"slhU": "up", "slhD": "down"}}
L6H = {"id": "L6H", "op": "ray", "in": "stem_rh", "out": {"srhU": "up", "srhD": "down"}}
STEM_EXT = ["asc_l", "asc_r", "desc_l", "desc_r"]
L6I = {"id": "L6I", "op": "conv", "in": ["run_v", "holeL", "holeR", "slhU", "slhD", "srhU", "srhD"], "out": STEM_EXT,
       "k": 1, "act": "clip01"}


def gen_L3R():
    section("L3S stroke edges: ink whose neighbour on one side is background")
    for name, (dy, dx) in EDGES.items():
        emit(f"TAP L3S {name} ink 0 0 1")
        emit(f"TAP L3S {name} ink {dy} {dx} -1", f"{name}: the stroke ends at this pixel on that side")
    section("L3R relational strokes: a run, minus/plus the rays just beside it")
    for name, (run, rays, bias, why) in REL.items():
        emit(f"TAP L3R {name} {run} 0 0 1")
        for ray, dy, dx, w in rays:
            emit(f"TAP L3R {name} {ray} {dy} {dx} {w}")
        emit(f"BIAS L3R {name} {bias}", why)


def gen_L4_L6():
    section("L4 outside seed: a background pixel with a clear ray to the border")
    for sfx, (rays, ink) in {"": (["inkL", "inkR", "inkU", "inkD"], "ink"),
                             "_f": (["fatL", "fatR", "fatU", "fatD"], "ink_fat"),
                             "_c": (["coreL", "coreR", "coreU", "coreD"], "ink_core")}.items():
        for r in rays:
            emit(f"TAP L4 outside{sfx} {r} 0 0 -1")
        emit(f"TAP L4 outside{sfx} {ink} 0 0 -4", "ink is never outside")
        emit(f"BIAS L4 outside{sfx} 4", "4 - (walls on all 4 sides) >= 1 when at least one ray is clear")
    section("L5 flood (looped 24x): outside spreads to 4-neighbours through background only")
    emit("TAP L5b outside outside_nb 0 0 1")
    emit("TAP L5b outside ink 0 0 -4", "ink blocks the flood")
    emit("TAP L5b outside_f outside_f_nb 0 0 1")
    emit("TAP L5b outside_f ink_fat 0 0 -4", "same flood on 1 px thicker ink: gaps of 1-2 px are sealed")
    emit("TAP L5b outside_c outside_c_nb 0 0 1")
    emit("TAP L5b outside_c ink_core 0 0 -4", "same flood with only near-white walls: grey loop centres stay open")
    section("L6 hole: background the flood never reached")
    emit("TAP L6 hole_x outside 0 0 -1")
    emit("TAP L6 hole_x ink 0 0 -4")
    emit("BIAS L6 hole_x 1", "hole_x = enclosed background (exact ink)")
    emit("TAP L6 hole_f outside_f 0 0 -1")
    emit("TAP L6 hole_f ink_fat 0 0 -4")
    emit("BIAS L6 hole_f 1", "hole_f = enclosed once small gaps are sealed: a loop the writer left slightly open")
    emit("TAP L6 hole_c outside_c 0 0 -1")
    emit("TAP L6 hole_c ink_core 0 0 -4")
    emit("BIAS L6 hole_c 1", "hole_c = enclosed by near-white ink: the small grey-centred loops of a thick 6, 8 or 9")
    emit("TAP L6B hole hole_c 0 0 1")
    emit("TAP L6B hole hole_x 0 0 1")
    emit("TAP L6B hole hole_f 0 0 1", "hole = either kind")
    emit("TAP L6D between_holes ink 0 0 1")
    emit("TAP L6D between_holes holeU 0 0 1")
    emit("TAP L6D between_holes holeD 0 0 1")
    emit("BIAS L6D between_holes -2", "ink with a loop above and a loop below it: the waist of 8, B, g")
    section("L6F stems beside a bowl (position-free)")
    emit("TAP L6F stem_lh run_v 0 0 1")
    emit("TAP L6F stem_lh holeR 0 0 1")
    emit("BIAS L6F stem_lh -1", "a vertical stroke with a hole somewhere to its right on the same row (b, p, D, P, B, R)")
    emit("TAP L6F stem_rh run_v 0 0 1")
    emit("TAP L6F stem_rh holeL 0 0 1")
    emit("BIAS L6F stem_rh -1", "a vertical stroke with a hole somewhere to its left on the same row (d, q, a, 9)")
    section("L6I a stem continuing past its bowl: ascenders (b, d) and descenders (p, q)")
    for name, ray, why in [("asc_l", "slhD", "above a stem that has its bowl to the right: the ascender of a b"),
                           ("asc_r", "srhD", "above a stem that has its bowl to the left: the ascender of a d"),
                           ("desc_l", "slhU", "below a stem that has its bowl to the right: the descender of a p"),
                           ("desc_r", "srhU", "below a stem that has its bowl to the left: the descender of a q")]:
        emit(f"TAP L6I {name} run_v 0 0 1")
        emit(f"TAP L6I {name} {ray} 0 0 1")
        emit(f"TAP L6I {name} holeL 0 0 -1")
        emit(f"TAP L6I {name} holeR 0 0 -1")
        emit(f"BIAS L6I {name} -1", f"a vertical stroke on a row with no hole, {why}")
    section("L6 bays: background walled on three sides, open on the fourth")
    for name, (w1, w2, w3, open_) in BAYS.items():
        for w in (w1, w2, w3):
            emit(f"TAP L6 {name} {w} 0 0 1")
        emit(f"TAP L6 {name} {open_} 0 0 -4", f"{name}: nothing on the open side")
        emit(f"TAP L6 {name} ink 0 0 -4")
        emit(f"TAP L6 {name} outside 0 0 0.5", "must be reachable from outside (not a hole)")
        emit(f"TAP L6 {name} outside_f 0 0 0.5", "and not inside a nearly closed loop")
        emit(f"BIAS L6 {name} -3", "three walls + outside twice = 4 -> 1")
    section("L6 gap: an empty row segment with ink above and below and nothing left or right (the dot of i, j)")
    emit("TAP L6 gap_v inkU 0 0 1")
    emit("TAP L6 gap_v inkD 0 0 1")
    emit("TAP L6 gap_v inkL 0 0 -4")
    emit("TAP L6 gap_v inkR 0 0 -4")
    emit("TAP L6 gap_v ink 0 0 -4")
    emit("BIAS L6 gap_v -1", "gap = the character is split into an upper and a lower piece")


# =============================================================================
# L7  zone features: how much of each map lies in each region
# =============================================================================
# Bands over the 28 px axis. MNIST digits sit in about 4..24, EMNIST characters in about 2..26.
ROWS = {"T": "0:10.5~1.5", "M": "10.5:17.5~1.5", "B": "17.5:28~1.5"}
COLS = {"l": "0:10.5~1.5", "c": "10.5:17.5~1.5", "r": "17.5:28~1.5"}
FULL = "0:28"
# (offset, scale): feature = clip((mass - offset) / scale)
SCALE = {"ink": (10.0, 10.0), "rel": (10.0, 8.0), "run": (15.0, 15.0), "end": (0.5, 1.5), "bay": (4.0, 8.0), "hole": (1.5, 3.0)}


def zones():
    z = {"A": (FULL, FULL)}
    for r, rs in ROWS.items():
        z[r] = (rs, FULL)
    for c, cs in COLS.items():
        z[c] = (FULL, cs)
    for r, rs in ROWS.items():
        for c, cs in COLS.items():
            z[r + c] = (rs, cs)
    return z


ZONES = zones()
FEATS = []           # (name, [maps], zone, kind)
for m in RUNS:
    FEATS += [(f"{m}@{z}", [m], z, "run") for z in ZONES]
for m in ENDS:
    FEATS += [(f"{m}@{z}", [m], z, "end") for z in ZONES]
FEATS += [(f"end_any@{z}", ENDS, z, "end") for z in ZONES]
for m in BAYS:
    FEATS += [(f"{m}@{z}", [m], z, "bay") for z in ZONES]
FEATS += [(f"hole@{z}", ["hole"], z, "hole") for z in ZONES]
FEATS += [(f"ink@{z}", ["ink"], z, "ink") for z in ZONES if len(z) == 2]
for m in REL:
    FEATS += [(f"{m}@{z}", [m], z, "rel") for z in ZONES]
# Scan-line features count LINES, not pixels: a row map is constant along its row, so the
# mass over a band of rows is 28 x (number of rows). Feature = clip((lines - 0.5) / 5).
SCAN_ROWS = dict(ROWS, A=FULL, U="0:14~1", D="14:28~1")
SCAN_COLS = dict(COLS, A=FULL)
SCAN = []            # (name, map, rows, cols, weight, bias)
for m in ["hx1", "hx2", "hx3"]:
    SCAN += [(f"{m}@{z}", m, rs, FULL, 1 / (28 * 5), -0.1) for z, rs in SCAN_ROWS.items()]
for m in ["vx1", "vx2", "vx3"]:
    SCAN += [(f"{m}@{z}", m, FULL, cs, 1 / (28 * 5), -0.1) for z, cs in SCAN_COLS.items()]
# Diagonal scan lines: share of the zone's pixels whose / (a) or \\ (b) line crosses k strokes.
# Feature = clip((share - 0.1) / 0.4): a quarter of the zone on such lines already counts strongly.
from charnet.engine import band as _band  # noqa: E402
for m in DIAG_MAPS:
    for z in ["A", "T", "M", "B", "l", "c", "r"]:
        rs, cs = ZONES[z]
        area = float(np.outer(_band(rs, 28), _band(cs, 28)).sum())
        SCAN += [(f"{m}@{z}", m, rs, cs, 1 / (area * 0.4), -0.25)]
# Stems beside a bowl, per band of rows: a stem of 8 rows already counts fully.
for m in ["stem_lh", "stem_rh"] + STEM_EXT:
    SCAN += [(f"{m}@{z}", m, ZONES[z][0], ZONES[z][1], 1 / 12, -0.15) for z in ["A", "T", "M", "B"]]
# Junctions and crossings: mass of the map in each zone (a meeting point is a few pixels across).
for m in ["junction", "crossing"]:
    SCAN += [(f"{m}@{z}", m, ZONES[z][0], ZONES[z][1], 1 / 6, -1 / 6) for z in ZONES]
# Profiles: background between the frame and the character, per band, in units of 60 px.
for m in ["marg_l", "marg_r"]:
    SCAN += [(f"{m}@{z}", m, rs, FULL, 1 / 60, 0) for z, rs in ROWS.items()]
for m in ["marg_t", "marg_b"]:
    SCAN += [(f"{m}@{z}", m, FULL, cs, 1 / 60, 0) for z, cs in COLS.items()]
TOP, BOT, LEFT, RIGHT = "0:14~2", "14:28~2", "0:14~2", "14:28~2"
EXTRA = [  # (name, [(map, rows, cols, weight)], offset, scale, why)
    ("loop", [("hole", FULL, FULL, 1)], 2, 4, "any loop, even a small one (MNIST loops can be a few pixels)"),
    ("loop_hi", [("hole", TOP, FULL, 1), ("hole", BOT, FULL, -1)], 2, 4, "loop sits in the upper half (9, P, e, q)"),
    ("loop_lo", [("hole", BOT, FULL, 1), ("hole", TOP, FULL, -1)], 2, 4, "loop sits in the lower half (6, b, d)"),
    ("loop_left", [("hole", FULL, LEFT, 1), ("hole", FULL, RIGHT, -1)], 6, 10, "loop sits to the left"),
    ("loop_right", [("hole", FULL, RIGHT, 1), ("hole", FULL, LEFT, -1)], 6, 10, "loop sits to the right"),
    ("hole_big", [("hole", FULL, FULL, 1)], 40, 20, "a large enclosed area (O, 0, D, Q); an 8's two loops rarely add up to this"),
    ("waist", [("between_holes", FULL, FULL, 1)], 1, 3, "a stroke with a loop above and below it"),
    ("wide_l", [("ink", FULL, "0:7~1", 1)], 2, 6, "ink reaches the left side"),
    ("wide_r", [("ink", FULL, "21:28~1", 1)], 2, 6, "ink reaches the right side"),
    ("gap", [("gap_v", FULL, FULL, 1)], 0.5, 2, "the character is split top/bottom (i, j)"),
]
L7_IN = (["ink"] + list(RUNS) + ENDS + list(BAYS) + ["hole", "gap_v", "between_holes"] + list(REL)
         + CROSS_MAPS + DIAG_MAPS + ["marg_l", "marg_r", "marg_t", "marg_b", "junction", "crossing", "stem_lh", "stem_rh"] + STEM_EXT)
L7 = {"id": "L7", "op": "zone", "in": L7_IN, "out": [f[0] for f in FEATS] + [e[0] for e in EXTRA] + [f[0] for f in SCAN],
      "act": "clip01"}


def gen_L7():
    section("L7 zone features: clip((mass in zone - offset) / scale)")
    for name, maps, z, kind in FEATS:
        off, sc = SCALE[kind]
        rs, cs = ZONES[z]
        for m in maps:
            emit(f"ZONE L7 {name} {m} {rs} {cs} {fmt(1 / sc)}")
        emit(f"BIAS L7 {name} {fmt(-off / sc)}")
    section("L7 shape features (a negative weight compares one region against another)")
    for name, terms, off, sc, why in EXTRA:
        for m, rs, cs, w in terms:
            emit(f"ZONE L7 {name} {m} {rs} {cs} {fmt(w / sc)}")
        emit(f"BIAS L7 {name} {fmt(-off / sc)}", why)
    section("L7 scan-line features: number of rows / columns with 1, 2, 3+ crossings; side profiles")
    for name, m, rs, cs, w, b in SCAN:
        emit(f"ZONE L7 {name} {m} {rs} {cs} {fmt(round(w, 6))}")
        emit(f"BIAS L7 {name} {fmt(b)}")


# =============================================================================
# L8 concepts, L9 one neuron per handwriting style, L10 class = its best style
# =============================================================================
from rules import (CONCEPTS, REF_GATE, REF_WEIGHT, REFEREES, STYLES, parse_terms,  # noqa: E402
                   style_bias, style_weights)

L7_OUT = L7["out"]
L8 = {"id": "L8", "op": "linear", "in": L7_OUT, "out": list(CONCEPTS), "act": "clip01"}
L9 = {"id": "L9", "op": "linear", "in": L7_OUT + list(CONCEPTS), "out": [f"s:{k}" for k in STYLES], "act": "none"}
L10 = {"id": "L10", "op": "maxout",
       "groups": {f"best:{c}": [f"s:{k}" for k in STYLES if k.split("/")[0] == c] for c in CLASSES}}
# Referees: for a pair of look-alike classes a/b, one hand-written judgement "a, not b" (v > 0) or
# "b, not a" (v < 0). A referee only speaks when the two classes are both in the running:
# its say is capped by gate = 1 - (top - min(best_a, best_b)) / REF_GATE.
PAIRS = [tuple(k.split(">")) for k in REFEREES]
assert len({frozenset(p) for p in PAIRS}) == len(PAIRS), "a pair of classes has two referees"
BEST = [f"best:{c}" for c in CLASSES]
L11 = {"id": "L11", "op": "linear", "in": L7_OUT + list(CONCEPTS),
       "out": [f"pro:{a}>{b}" for a, b in PAIRS] + [f"pro:{b}>{a}" for a, b in PAIRS], "act": "clip01"}
L11G = {"id": "L11G", "op": "linear", "in": BEST, "out": [f"gap:{a}>{b}" for a, b in PAIRS] + [f"gap:{b}>{a}" for a, b in PAIRS],
        "act": "relu"}
L11T = {"id": "L11T", "op": "maxout", "groups": {"top": BEST}}
L12 = {"id": "L12", "op": "linear", "in": L11G["out"] + BEST + ["top"], "out": [f"gate:{a}/{b}" for a, b in PAIRS],
       "act": "clip01"}
L13 = {"id": "L13", "op": "linear", "in": L11["out"] + L12["out"], "out": [f"cut:{a}>{b}" for a, b in PAIRS] +
       [f"cut:{b}>{a}" for a, b in PAIRS], "act": "relu"}
DIGITS = [c for c in CLASSES if c.isdigit()]
LETTERS = [c for c in CLASSES if c.isalpha()]
# cls:* ranks all 62 with every referee; dig:* and let:* rank one kind only, heard only by referees within that kind
# (once the writer has said "a digit", a referee about a digit and a letter has nothing to add).
# L13S each referee's verdict as one neuron: say:x>y = min(pro, gate) = pro - cut (0 when it is silent).
# L13M a class is pushed down by its STRONGEST single referee loss, not the sum: several referees each saying
# "not a, a little" (a class that looks like many others has many referees) must not add up to "certainly not a".
SAYS = [f"say:{x}>{y}" for a, b in PAIRS for x, y in ((a, b), (b, a))]
L13S = {"id": "L13S", "op": "linear", "in": L11["out"] + L13["out"], "out": SAYS + ["zero"], "act": "relu"}
L13M = {"id": "L13M", "op": "maxout", "groups": {
    **{f"loss:{c}": [f"say:{x}>{y}" for a, b in PAIRS for x, y in ((a, b), (b, a)) if y == c] + ["zero"] for c in CLASSES},
    **{f"losk:{c}": [f"say:{x}>{y}" for a, b in PAIRS for x, y in ((a, b), (b, a)) if y == c and x.isdigit() == c.isdigit()]
       + ["zero"] for c in CLASSES}}}
L14 = {"id": "L14", "op": "linear", "in": BEST + [f"loss:{c}" for c in CLASSES] + [f"losk:{c}" for c in CLASSES],
       "out": [f"cls:{c}" for c in CLASSES] + [f"dig:{c}" for c in DIGITS] + [f"let:{c}" for c in LETTERS], "act": "none"}


def gen_readout():
    # every name a rule uses must exist, or the edit list would not load
    for name, (terms, bias, why) in CONCEPTS.items():
        for f, w in parse_terms(terms):
            assert f in L7_OUT, f"concept {name}: {f} is not a zone feature (concepts read L7 only)"
    for k, rule in STYLES.items():
        for f, w in style_weights(rule):
            assert f in L9["in"], f"style {k}: unknown feature {f}"
    for key, (terms, bias, why) in REFEREES.items():
        for f, w in parse_terms(terms):
            assert f in L11["in"], f"referee {key}: unknown feature {f}"
    section("L8 concepts")
    for name, (terms, bias, why) in CONCEPTS.items():
        for f, w in parse_terms(terms):
            emit(f"LINK L8 {name} {f} {fmt(w)}")
        emit(f"BIAS L8 {name} {fmt(bias)}", why)
    section("L9 styles: positive evidence sums to 1, each contradiction subtracts")
    for k, rule in STYLES.items():
        for f, w in style_weights(rule):
            emit(f"LINK L9 s:{k} {f} {fmt(round(w, 4))}")
        emit(f"BIAS L9 s:{k} {fmt(style_bias(rule))}", f"{k}: {rule}")
    section("L11 referees: v = evidence for a minus evidence for b;  pro:a>b = clip(v), pro:b>a = clip(-v)")
    for key, (terms, bias, why) in REFEREES.items():
        a, b = key.split(">")
        for f, w in parse_terms(terms):
            emit(f"LINK L11 pro:{a}>{b} {f} {fmt(w)}")
            emit(f"LINK L11 pro:{b}>{a} {f} {fmt(-w)}")
        emit(f"BIAS L11 pro:{a}>{b} {fmt(bias)}", f"{a} vs {b}: {why}")
        emit(f"BIAS L11 pro:{b}>{a} {fmt(-bias)}")
    section(f"L11G-L13 gate: a referee speaks only while both classes are within {REF_GATE} of the top score")
    for a, b in PAIRS:
        emit(f"LINK L11G gap:{a}>{b} best:{a} 1")
        emit(f"LINK L11G gap:{a}>{b} best:{b} -1", "how far a leads b")
        emit(f"LINK L11G gap:{b}>{a} best:{b} 1")
        emit(f"LINK L11G gap:{b}>{a} best:{a} -1")
        # min(best_a, best_b) = best_a - gap:a>b; gate = 1 - (top - min) / REF_GATE
        emit(f"LINK L12 gate:{a}/{b} top {fmt(-1 / REF_GATE)}")
        emit(f"LINK L12 gate:{a}/{b} best:{a} {fmt(1 / REF_GATE)}")
        emit(f"LINK L12 gate:{a}/{b} gap:{a}>{b} {fmt(-1 / REF_GATE)}")
        emit(f"BIAS L12 gate:{a}/{b} 1", "gate = 1 - (top score - weaker of the pair) / REF_GATE: both must be in the running")
        for x, y in ((a, b), (b, a)):
            emit(f"LINK L13 cut:{x}>{y} pro:{x}>{y} 1")
            emit(f"LINK L13 cut:{x}>{y} gate:{a}/{b} -1", "the part of the judgement above the gate (removed next)")
    section("L13S referee verdicts: say = pro - cut (the part of the judgement under the gate)")
    for a, b in PAIRS:
        for x, y in ((a, b), (b, a)):
            emit(f"LINK L13S say:{x}>{y} pro:{x}>{y} 1")
            emit(f"LINK L13S say:{x}>{y} cut:{x}>{y} -1", f"{x} over {y}: how strongly, while both are in the running")
    section(f"L14 class score = best style - {2 * REF_WEIGHT} x (its strongest referee loss)")
    # A referee judges one pair. It pushes the loser down and leaves the winner where it was (lifting winners let
    # a digit that won several pairs jump over letters: +1.18 on average for the best digit on a letter image), and
    # only the strongest loss counts (summed losses pushed a's down by 1.67 on average where their own rule scored
    # higher than the winner's).
    for c in CLASSES:
        k = "dig" if c.isdigit() else "let"
        emit(f"LINK L14 cls:{c} best:{c} 1")
        emit(f"LINK L14 cls:{c} loss:{c} {fmt(-2 * REF_WEIGHT)}", f"{c}: its strongest referee loss among all 62")
        emit(f"LINK L14 {k}:{c} best:{c} 1")
        emit(f"LINK L14 {k}:{c} losk:{c} {fmt(-2 * REF_WEIGHT)}", f"{c}: its strongest loss to its own kind")

LAYERS = [L1, L3F, L2, L3, LX1, LX2, LX3, LX4, LX5, LD1, LD2, LD3, LD4, LD5, LJ0, LJ1, LJ2, LJ3, L3S, L3T, L3U, L3V, L3W, L3R, L3G, L3H, L4, L5, L6, L6B, L6C, L6D, L6E, L6F, L6G, L6H, L6I, L7, L8, L9, L10, L11, L11G, L11T, L12, L13, L13S, L13M, L14]


def build():
    lines.clear()
    lines.append("# WP-CharNet weights, programmed directly: every nonzero parameter comes from a line below")
    lines.append("# No training, no fine-tuning, no weights or statistics taken from any dataset.")
    gen_L1()
    gen_L2()
    gen_L3R()
    gen_LX()
    gen_LD()
    gen_LJ()
    gen_L4_L6()
    gen_L7()
    gen_readout()
    arch = {"name": "WP-CharNet", "input": [28, 28], "classes": CLASSES,
            "layers": LAYERS}
    return arch, "\n".join(lines) + "\n"


if __name__ == "__main__":
    arch, edits = build()
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    with open(os.path.join(ROOT, "arch.json"), "w") as f:
        json.dump(arch, f, indent=1)
    with open(os.path.join(ROOT, "weights", "charnet.edits"), "w") as f:
        f.write(edits)
    n = sum(1 for l in edits.splitlines() if l.strip() and not l.startswith("#"))
    print(f"wrote arch.json and weights/charnet.edits ({n} edit lines)")
