"""Per-neuron tests: each named map must fire on drawings that have its feature and stay
quiet on drawings that don't. Drawings are clean (no writer variation) author-drawn strokes."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from charnet.engine import load  # noqa: E402
from gen.handwriting import A, L, normalize, render  # noqa: E402
from gen.generate import ENDS  # noqa: E402

NET = load(os.path.join(ROOT, "arch.json"), os.path.join(ROOT, "weights", "charnet.edits"))


def run(*strokes, mode="mnist"):
    img = normalize(render(list(strokes), np.random.default_rng(0), var=0.0), mode)
    return NET.forward(img)


def mass(T, name):
    return float(T[name].sum())


BAR = L((0, .5), (1, .5))
STEM = L((.5, 0), (.5, 1))
RING = A(.5, .5, .5, .5, 90, 450)
CUP = L((0, 0), (0, 1), (1, 1), (1, 0))          # U shape: open at the top


def test_every_parameter_comes_from_the_edit_list():
    from charnet.engine import Net
    import json
    with open(os.path.join(ROOT, "arch.json")) as f:
        blank = Net(json.load(f))
    assert all(not np.any(p) for d in blank.P.values() for p in d.values())
    assert NET.counts()[1] > 0


def test_ink_is_binary_ish():
    T = run(STEM)
    assert T["ink"].max() == 1.0 and T["ink"].min() == 0.0


@pytest.mark.parametrize("strokes,on,off", [
    ([BAR], "run_h", "run_v"),
    ([STEM], "run_v", "run_h"),
    ([L((0, 1), (1, 0))], "run_d1", "run_d2"),
    ([L((0, 0), (1, 1))], "run_d2", "run_d1"),
])
def test_runs_follow_direction(strokes, on, off):
    T = run(*strokes)
    assert mass(T, on) > 5 * max(mass(T, off), 1)


def test_stem_has_ends_up_and_down_only():
    T = run(STEM)
    assert mass(T, "end_up") > 0.5 and mass(T, "end_down") > 0.5
    assert mass(T, "end_left") + mass(T, "end_right") < 0.5


def test_closed_ring_has_no_ends_and_a_hole():
    T = run(RING)
    assert sum(mass(T, e) for e in ENDS) < 0.5
    assert mass(T, "hole") > 30


def test_slightly_open_ring_still_has_a_hole():
    T = run(A(.5, .5, .5, .5, 100, 440))       # 20 degree gap at the top
    assert mass(T, "hole") > 20


def test_c_is_a_bay_open_to_the_right_not_a_hole():
    T = run(A(.5, .5, .5, .5, 50, 310))
    assert mass(T, "hole") < 2
    assert mass(T, "bay_R") > 20 and mass(T, "bay_L") < 2


def test_cup_opens_up():
    T = run(CUP)
    assert mass(T, "bay_U") > 20 and mass(T, "bay_D") < 2 and mass(T, "hole") < 2


def test_gap_fires_for_a_dot_over_a_stem():
    T = run(L((.5, .35), (.5, 1)), A(.5, .1, .04, .04, 0, 360, 10))
    assert mass(T, "gap_v") > 0.5
    assert mass(run(STEM), "gap_v") < 0.5


def test_between_holes_fires_for_stacked_loops_only():
    eight = run(A(.5, .25, .4, .25, -90, 270), A(.5, .75, .45, .25, 90, 450))
    assert mass(eight, "between_holes") > 1
    assert mass(run(RING), "between_holes") < 1


def test_relational_strokes():
    T = run(L((0, 0), (0, 1), (.6, 1)))          # L
    assert T["edge_l"].sum() > 10                # the stem has nothing to its left
    assert T["edge_t"][18:].sum() > 5            # the foot has nothing above it
    E = run(L((.6, 0), (0, 0), (0, 1), (.6, 1)), L((0, .5), (.5, .5)))
    assert E["mid_h"][10:18].sum() > 5           # E's middle bar has ink above and below


def rows_with(T, m):
    """Number of rows (h maps) or columns (v maps) where map m is on."""
    return float(T[m].sum() / 28)


def test_ring_rows_cross_two_strokes():
    T = run(RING)
    assert rows_with(T, "hx2") > 10 and rows_with(T, "hx3") < 1
    assert T["cnt_h"][14, 0] == pytest.approx(2, abs=0.3)


def test_eight_reads_two_one_two():
    T = run(A(.5, .25, .4, .25, -90, 270), A(.5, .75, .45, .25, 90, 450))
    c = T["cnt_h"][:, 0]
    rows = [r for r in range(28) if c[r] > 0.25]
    top, bottom = rows[0], rows[-1]
    third = (bottom - top) // 4
    assert round(c[top + third]) == 2 and round(c[bottom - third]) == 2
    assert min(round(c[r]) for r in range((top + bottom) // 2 - 1, (top + bottom) // 2 + 2)) == 1


def test_e_columns_cross_three_bars():
    T = run(L((.6, 0), (0, 0), (0, 1), (.6, 1)), L((0, .5), (.5, .5)))
    assert rows_with(T, "vx3") > 3
    assert rows_with(run(STEM), "vx3") < 0.5


def test_profiles():
    T = run(L((.5, 0), (.5, 1)))                  # centred stem: margins on both sides, none above/below it
    assert T["marg_l"][14].sum() > 5 and T["marg_r"][14].sum() > 5
    assert T["marg_l"][14].sum() == pytest.approx(T["marg_r"][14].sum(), abs=3)


def test_branches_count_strokes_leaving_a_point():
    X_ = run(L((0, 0), (1, 1)), L((1, 0), (0, 1)))
    T_ = run(L((0, 0), (1, 0)), L((.5, 0), (.5, 1)))
    assert mass(X_, "crossing") > 1 and mass(X_, "junction") > 1
    assert mass(T_, "junction") > 1 and mass(T_, "crossing") < 0.5
    for plain in (STEM, RING, A(.5, .5, .5, .5, 50, 310)):
        T = run(plain)
        assert mass(T, "junction") < 0.5 and mass(T, "crossing") < 0.5


def test_stem_tip_has_one_branch():
    T = run(STEM)
    ys, xs = np.nonzero(T["ink"] > 0.99)
    top = ys.min()
    assert T["branches"][top + 1, int(np.median(xs[ys == top]))] == pytest.approx(1, abs=0.5)


def test_diagonal_scan_lines():
    """A / stroke is crossed once by the \\ lines through it; an X's centre \\ and / lines each see its other stroke;
    a V's / lines through its middle cross both arms."""
    from gen.handwriting import render, normalize, L
    net = NET
    slash = normalize(render([L((0.8, 0), (0.2, 1))], var=0), "mnist")
    T = net.forward(slash)
    assert T["bx1@M"] > 0.5 and T["bx2@M"] < 0.2
    vee = normalize(render([L((0, 0), (0.35, 1), (0.7, 0))], var=0), "mnist")
    T = net.forward(vee)
    assert T["ax2@T"] > 0.3 or T["bx2@T"] > 0.3


def test_stem_beside_bowl():
    """b: a stem with the bowl to its right, only low down; d: the mirror. Position-free (the letters are centred)."""
    from gen.handwriting import render, normalize, L, A, cat
    b = normalize(render([cat(L((0, 0), (0, 1)), A(.25, .72, .25, .28, 180, -180))], var=0), "mnist")
    T = NET.forward(b)
    assert T["stem_lh@B"] > 0.5 and T["stem_lh@T"] < 0.3
    assert T["asc_l@A"] > 0.5 and T["asc_r@A"] < 0.4
    d = normalize(render([cat(L((.5, 0), (.5, 1)), A(.25, .72, .25, .28, 0, 360))], var=0), "mnist")
    T = NET.forward(d)
    assert T["stem_rh@B"] > 0.5 and T["stem_rh@T"] < 0.3
    assert T["asc_r@A"] > 0.5 and T["asc_l@A"] < 0.4
