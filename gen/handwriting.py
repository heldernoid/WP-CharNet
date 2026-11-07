"""Author-drawn development set: every character as pen strokes, in several styles,
rendered with per-writer variation. This is the only data used while authoring weights.
(A quick check on clean drawings; the rules were written from MNIST / EMNIST training images, see LOG.md.)

Coordinates: x to the right, y down, the character roughly inside a 1-high box.
Angles in degrees, counter-clockwise on screen (0 = right, 90 = up).
"""
import math
import numpy as np
from PIL import Image, ImageDraw


def L(*pts):
    return [tuple(p) for p in pts]


def A(cx, cy, rx, ry, a0, a1, n=None):
    n = n or max(8, int(abs(a1 - a0) / 8))
    return [(cx + rx * math.cos(math.radians(a)), cy - ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


def Bz(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    P = [np.array(p) for p in (p0, p1, p2, p3)]
    pts = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
    return [tuple(p) for p in pts]


def cat(*parts):
    out = []
    for p in parts:
        out += p if not out or out[-1] != p[0] else p[1:]
    return out


def dot(x, y, r=0.035):
    return A(x, y, r, r, 0, 360, 10)


# Each entry: char -> {style: [strokes]}. Styles are real alternatives people use.
G = {
    "0": {"oval": [A(.3, .5, .3, .5, 90, 450)]},   # slashed zero: not supported yet (reads as 8)
    "1": {"stick": [L((.3, 0), (.3, 1))],
          "flag": [L((.08, .22), (.3, 0), (.3, 1))],
          "flag_base": [L((.08, .22), (.3, 0), (.3, 1)), L((.08, 1), (.52, 1))]},
    "2": {"curve": [cat(A(.3, .28, .28, .26, 160, -30), L((.54, .41), (.02, 1), (.62, 1)))],
          "loop": [cat(A(.3, .28, .28, .26, 160, -30), L((.54, .41), (.05, .95)),
                       A(.12, .9, .08, .08, 270, 450), L((.12, .98), (.62, 1)))]},
    "3": {"round": [A(.28, .25, .27, .25, 150, -90), A(.28, .74, .3, .26, 90, -150)],
          "flat_top": [cat(L((.02, 0), (.55, 0), (.22, .4)), A(.25, .7, .3, .3, 90, -150))]},
    "4": {"closed": [L((.45, 1), (.45, 0), (0, .68), (.65, .68))],
          "open": [L((.08, 0), (.02, .62), (.65, .62)), L((.48, .25), (.48, 1))]},
    "5": {"std": [L((.58, 0), (.08, 0), (.05, .45)), A(.28, .68, .3, .32, 137, -150)]},
    "6": {"std": [cat(Bz((.5, 0), (.1, .08), (0, .45), (.03, .7)), A(.3, .7, .27, .28, 180, -180))]},
    "7": {"plain": [L((0, 0), (.6, 0), (.2, 1))],
          "crossed": [L((0, 0), (.6, 0), (.2, 1)), L((.15, .5), (.55, .5))]},
    "8": {"stacked": [A(.3, .25, .24, .25, -90, 270), A(.3, .74, .29, .26, 90, 450)]},
    "9": {"straight": [A(.3, .28, .28, .28, 0, 360), L((.58, .28), (.52, 1))],
          "curved": [A(.3, .28, .28, .28, 0, 360), Bz((.58, .28), (.58, .7), (.45, 1), (.1, .95))]},
    "A": {"std": [L((0, 1), (.35, 0), (.7, 1)), L((.15, .6), (.55, .6))]},
    "B": {"std": [cat(L((0, 1), (0, 0), (.35, 0)), A(.35, .24, .22, .24, 90, -90), L((.35, .48), (.4, .48)),
                      A(.4, .74, .25, .26, 90, -90), L((.4, 1), (0, 1)))]},
    "C": {"std": [A(.4, .5, .4, .5, 50, 310)]},
    "D": {"std": [cat(L((0, 1), (0, 0), (.25, 0)), A(.25, .5, .4, .5, 90, -90), L((.25, 1), (0, 1)))]},
    "E": {"std": [L((.55, 0), (0, 0), (0, 1), (.55, 1)), L((0, .5), (.45, .5))]},
    "F": {"std": [L((.55, 0), (0, 0), (0, 1)), L((0, .48), (.45, .48))]},
    "G": {"std": [cat(A(.4, .5, .4, .5, 45, 315), L((.68, .85), (.7, .55), (.42, .55)))],
          "no_bar": [A(.4, .5, .4, .5, 45, 330), L((.75, .55), (.45, .55))]},
    "H": {"std": [L((0, 0), (0, 1)), L((.6, 0), (.6, 1)), L((0, .5), (.6, .5))]},
    "I": {"stick": [L((.15, 0), (.15, 1))],
          "serif": [L((.15, 0), (.15, 1)), L((0, 0), (.3, 0)), L((0, 1), (.3, 1))]},
    "J": {"hook": [cat(L((.55, 0), (.55, .72)), A(.3, .72, .25, .28, 0, -180))],
          "bar": [cat(L((.55, 0), (.55, .72)), A(.3, .72, .25, .28, 0, -180)), L((.3, 0), (.8, 0))]},
    "K": {"std": [L((0, 0), (0, 1)), L((.55, 0), (0, .55)), L((.18, .4), (.6, 1))],
          "arrow": [L((0, 0), (0, 1)), L((.55, 0), (0, .5), (.6, 1))]},
    "L": {"std": [L((0, 0), (0, 1), (.5, 1))]},
    "M": {"std": [L((0, 1), (.05, 0), (.4, .65), (.75, 0), (.8, 1))],
          "short_v": [L((0, 1), (0, 0), (.4, .45), (.8, 0), (.8, 1))]},
    "N": {"std": [L((0, 1), (0, 0), (.6, 1), (.6, 0))]},
    "O": {"std": [A(.4, .5, .4, .5, 90, 450)]},
    "P": {"std": [cat(L((0, 1), (0, 0), (.3, 0)), A(.3, .25, .25, .25, 90, -90), L((.3, .5), (0, .5)))]},
    "Q": {"std": [A(.4, .5, .4, .5, 90, 450), L((.45, .7), (.85, 1.05))]},
    "R": {"std": [cat(L((0, 1), (0, 0), (.3, 0)), A(.3, .25, .25, .25, 90, -90), L((.3, .5), (0, .5))),
                  L((.2, .5), (.6, 1))]},
    "S": {"std": [cat(A(.3, .25, .28, .25, 30, 270), A(.3, .75, .3, .25, 90, -150))]},
    "T": {"std": [L((0, 0), (.7, 0)), L((.35, 0), (.35, 1))]},
    "U": {"std": [cat(L((0, 0), (0, .65)), A(.3, .65, .3, .35, 180, 360), L((.6, .65), (.6, 0)))]},
    "V": {"std": [L((0, 0), (.35, 1), (.7, 0))]},
    "W": {"std": [L((0, 0), (.22, 1), (.45, .3), (.68, 1), (.9, 0))]},
    "X": {"std": [L((0, 0), (.65, 1)), L((.65, 0), (0, 1))]},
    "Y": {"std": [L((0, 0), (.35, .5), (.7, 0)), L((.35, .5), (.35, 1))],
          "one_stroke": [L((0, 0), (.35, .5)), L((.7, 0), (.1, 1))]},
    "Z": {"std": [L((0, 0), (.6, 0), (0, 1), (.6, 1))],
          "crossed": [L((0, 0), (.6, 0), (0, 1), (.6, 1)), L((.12, .5), (.48, .5))]},
    # lowercase: drawn in the box the letter actually occupies (EMNIST rescales each letter)
    "a": {"round": [A(.27, .5, .27, .48, 0, 360), L((.54, .05), (.56, 1))],
          "storey": [cat(A(.25, .25, .22, .2, 150, 0), L((.47, .25), (.47, 1))),
                     cat(L((.47, .55), (.22, .52)), A(.22, .76, .22, .22, 90, 270), L((.22, .98), (.47, .85)))]},
    "b": {"std": [L((0, 0), (0, 1)), A(.27, .72, .27, .28, 180, 540)]},
    "c": {"std": [A(.3, .5, .3, .5, 50, 310)]},
    "d": {"std": [A(.27, .72, .27, .28, 0, 360), L((.54, 0), (.54, 1))]},
    "e": {"std": [cat(L((.02, .5), (.6, .5)), A(.31, .5, .29, .5, 0, 320))]},
    "f": {"std": [cat(Bz((.55, .05), (.35, -.05), (.2, 0), (.2, .25)), L((.2, .25), (.2, 1))),
                  L((0, .4), (.45, .4))]},
    "g": {"nine": [A(.27, .28, .27, .28, 0, 360), cat(L((.54, .05), (.54, .8)), A(.3, .8, .24, .2, 0, -160))],
          "loop": [A(.27, .28, .27, .28, 0, 360),
                   cat(L((.54, .05), (.54, .7)), Bz((.54, .7), (.5, 1.05), (0, 1.05), (.1, .75)),
                       L((.1, .75), (.54, .62)))]},
    "h": {"std": [L((0, 0), (0, 1)), cat(Bz((0, .62), (.1, .4), (.5, .35), (.5, .6)), L((.5, .6), (.5, 1)))]},
    "i": {"std": [L((.1, .35), (.1, 1)), dot(.1, .1)]},
    "j": {"std": [cat(L((.35, .35), (.35, .85)), A(.15, .85, .2, .15, 0, -150)), dot(.35, .1)]},
    "k": {"std": [L((0, 0), (0, 1)), L((.45, .4), (0, .72)), L((.15, .62), (.5, 1))]},
    "l": {"stick": [L((.1, 0), (.1, 1))],
          "tail": [cat(L((.1, 0), (.1, .85)), Bz((.1, .85), (.1, 1), (.2, 1), (.3, .95)))]},
    "m": {"std": [L((0, 0), (0, 1)),
                  cat(Bz((0, .25), (.05, 0), (.45, -.05), (.47, .3)), L((.47, .3), (.47, 1))),
                  cat(Bz((.47, .3), (.5, 0), (.95, -.05), (.95, .3)), L((.95, .3), (.95, 1)))]},
    "n": {"std": [L((0, 0), (0, 1)), cat(Bz((0, .3), (.05, 0), (.55, -.05), (.55, .35)), L((.55, .35), (.55, 1)))]},
    "o": {"std": [A(.3, .5, .3, .5, 90, 450)]},
    "p": {"std": [L((0, 0), (0, 1)), A(.27, .3, .27, .3, 180, 540)]},
    "q": {"std": [A(.27, .3, .27, .3, 0, 360), L((.54, 0), (.54, 1))],
          "hook": [A(.27, .3, .27, .3, 0, 360), cat(L((.54, 0), (.54, .9)), L((.54, .9), (.72, .78)))]},
    "r": {"std": [L((0, 0), (0, 1)), Bz((0, .35), (.1, .05), (.3, 0), (.5, .08))]},
    "s": {"std": [cat(A(.3, .25, .28, .25, 30, 270), A(.3, .75, .3, .25, 90, -150))]},
    "t": {"std": [cat(L((.25, 0), (.25, .85)), Bz((.25, .85), (.27, 1), (.4, 1), (.5, .93))), L((0, .3), (.5, .3))]},
    "u": {"std": [cat(L((0, 0), (0, .65)), A(.27, .65, .27, .35, 180, 360)), L((.55, 0), (.55, 1))]},
    "v": {"std": [L((0, 0), (.35, 1), (.7, 0))]},
    "w": {"std": [L((0, 0), (.22, 1), (.45, .3), (.68, 1), (.9, 0))],
          "round": [cat(L((0, 0), (0, .7)), A(.22, .7, .22, .3, 180, 360), L((.44, .7), (.44, .35)),
                        L((.44, .7)), A(.66, .7, .22, .3, 180, 360), L((.88, .7), (.88, 0)))]},
    "x": {"std": [L((0, 0), (.65, 1)), L((.65, 0), (0, 1))]},
    "y": {"std": [L((0, 0), (.3, .55)), L((.6, 0), (.15, 1))],
          "u_tail": [cat(L((0, 0), (0, .35)), A(.27, .35, .27, .25, 180, 360)),
                     cat(L((.54, 0), (.54, .8)), A(.3, .8, .24, .2, 0, -160))]},
    "z": {"std": [L((0, 0), (.6, 0), (0, 1), (.6, 1))]},
}

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
assert sorted(G) == sorted(CLASSES)


# ---- per-writer variation ---------------------------------------------------
def _wobble(pts, rng, amp):
    """Low-frequency hand tremor along a stroke."""
    p = np.array(pts, dtype=float)
    if len(p) < 2 or amp == 0:
        return p
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    f = rng.uniform(1.5, 4.0, size=2)
    ph = rng.uniform(0, 2 * np.pi, size=2)
    return p + amp * np.stack([np.sin(f[0] * 2 * np.pi * s + ph[0]), np.sin(f[1] * 2 * np.pi * s + ph[1])], 1)


def _trim(pts, rng, frac):
    """Open a loop a little (gap) or overshoot the end, as fast writers do."""
    p = np.array(pts, dtype=float)
    if len(p) < 6 or frac == 0:
        return p
    closed = np.linalg.norm(p[0] - p[-1]) < 0.05
    if not closed:
        return p
    k = int(len(p) * frac)
    if rng.random() < 0.5:
        return p[: len(p) - k]                       # gap
    return np.concatenate([p, p[1:k + 1]])           # overshoot


def render(strokes, rng=None, var=1.0, size=112):
    """Draw strokes with random writer variation; returns a size x size grey image in [0, 1]."""
    rng = rng or np.random.default_rng(0)
    slant = rng.uniform(-0.35, 0.25) * var
    rot = math.radians(rng.uniform(-7, 7) * var)
    sx = math.exp(rng.uniform(-0.25, 0.25) * var)
    width = rng.uniform(0.07, 0.17)                   # pen width as a fraction of character height
    amp = rng.uniform(0, 0.03) * var
    trim = rng.uniform(0, 0.07) * var
    c, s = math.cos(rot), math.sin(rot)
    out = []
    for st in strokes:
        p = _trim(st, rng, trim)
        p = _wobble(p, rng, amp)
        x, y = p[:, 0] * sx, p[:, 1]
        x = x - slant * (y - 0.5)                     # positive slant leans right (y is down)
        x, y = c * x - s * (y - .5), s * x + c * (y - .5) + .5
        out.append(np.stack([x, y], 1))
    allp = np.concatenate(out)
    lo, hi = allp.min(0), allp.max(0)
    scale = (size * 0.7) / max(hi[1] - lo[1], hi[0] - lo[0], 1e-6)
    off = size / 2 - (lo + hi) / 2 * scale
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    w = max(1, int(round(width * (hi[1] - lo[1]) * scale)))
    for p in out:
        q = [tuple(v) for v in p * scale + off]
        if len(q) > 1:
            d.line(q, fill=255, width=w)
        for v in q:
            d.ellipse([v[0] - w / 2, v[1] - w / 2, v[0] + w / 2, v[1] + w / 2], fill=255)
    return np.asarray(img, dtype=float) / 255.0


def normalize(img, mode="mnist"):
    """Bring a drawing to 28x28 the way the datasets did (as described in their papers).

    mnist : fit the bounding box into 20x20 keeping aspect, centre of mass at the centre.
    emnist: fit the bounding box into 24x24 keeping aspect (2 px border), box centred.
    The browser demo runs the same steps on the canvas.
    """
    ys, xs = np.nonzero(img > 0.05)
    if len(ys) == 0:
        return np.zeros((28, 28))
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    box = 20 if mode == "mnist" else 24
    h, w = crop.shape
    f = box / max(h, w)
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS),
                       dtype=float) / 255.0
    out = np.zeros((28, 28))
    if mode == "mnist":
        tot = small.sum() or 1
        cy = (small.sum(1) * np.arange(nh)).sum() / tot
        cx = (small.sum(0) * np.arange(nw)).sum() / tot
        oy, ox = int(round(13.5 - cy)), int(round(13.5 - cx))
    else:
        oy, ox = (28 - nh) // 2, (28 - nw) // 2
    oy, ox = min(max(oy, 0), 28 - nh), min(max(ox, 0), 28 - nw)
    out[oy:oy + nh, ox:ox + nw] = small
    return np.clip(out, 0, 1)


def sample(ch, style=None, seed=0, var=1.0, mode="mnist"):
    rng = np.random.default_rng(seed)
    styles = G[ch]
    style = style or list(styles)[rng.integers(len(styles))]
    return normalize(render(styles[style], rng, var), mode), style


def devset(n_per_style=6, seed=1, modes=("mnist", "emnist")):
    """Yields (image, char, style, mode). Deterministic."""
    k = 0
    for ch in CLASSES:
        for style in G[ch]:
            for mode in modes:
                for i in range(n_per_style):
                    img, _ = sample(ch, style, seed=seed * 100003 + k, mode=mode)
                    k += 1
                    yield img, ch, style, mode
