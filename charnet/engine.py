"""WP-CharNet engine: build parameters from the edit list, run the forward pass.

All parameters start at zero. `apply_edits` writes the edit list into them.
Edit kinds (one per line, `#` starts a comment):

  TAP   <layer> <out> <in> <dy> <dx> <value>      conv kernel tap (dy, dx relative to the centre)
  BIAS  <layer> <out> <value>                     bias of an output channel / feature / neuron
  ZONE  <layer> <feat> <in> <rows> <cols> <scale> pooling mask over one input map
          rows/cols: a:b or a:b~w (full on [a, b), linear ramp of width w on both sides)
  LINK  <layer> <out> <in> <value>                one weight of a linear layer

Maps are 28x28 float arrays. Features and neurons are scalars. Every intermediate
value is kept in the returned trace, so any activation can be shown and explained.
"""
import json
import numpy as np

H = W = 28


def band(spec, n):
    """Trapezoid mask along one axis: 'a:b' or 'a:b~w'."""
    ramp = 0.0
    if "~" in spec:
        spec, r = spec.split("~")
        ramp = float(r)
    a, b = (float(v) for v in spec.split(":"))
    c = np.arange(n) + 0.5
    m = np.ones(n)
    if ramp > 0:
        m = np.minimum(m, np.clip((c - (a - ramp)) / ramp, 0, 1))
        m = np.minimum(m, np.clip(((b + ramp) - c) / ramp, 0, 1))
    else:
        m = ((c >= a) & (c < b)).astype(float)
    return m


ACT = {
    "relu": lambda v: np.maximum(v, 0),
    "clip01": lambda v: np.clip(v, 0, 1),
    "none": lambda v: v,
}


class Net:
    def __init__(self, arch):
        self.arch = arch
        self.P = {}
        for L in self._all_layers(arch["layers"]):
            op, lid = L["op"], L["id"]
            if op == "conv":
                k = L["k"]
                self.P[lid] = {"W": np.zeros((len(L["out"]), len(L["in"]), k, k)),
                               "b": np.zeros(len(L["out"]))}
            elif op == "zone":
                self.P[lid] = {"W": np.zeros((len(L["out"]), len(L["in"]), H, W)),
                               "b": np.zeros(len(L["out"]))}
            elif op == "linear":
                self.P[lid] = {"W": np.zeros((len(L["out"]), len(L["in"]))),
                               "b": np.zeros(len(L["out"]))}
        self.L = {L["id"]: L for L in self._all_layers(arch["layers"])}

    @staticmethod
    def _all_layers(layers):
        for L in layers:
            if L["op"] == "loop":
                yield from Net._all_layers(L["body"])
            else:
                yield L

    # ---- edits -------------------------------------------------------------
    def apply_edits(self, text):
        for n, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            t = line.split()
            kind, lid = t[0], t[1]
            L, P = self.L[lid], self.P[lid]
            try:
                if kind == "TAP":
                    o, i = L["out"].index(t[2]), L["in"].index(t[3])
                    c = L["k"] // 2
                    P["W"][o, i, c + int(t[4]), c + int(t[5])] = float(t[6])
                elif kind == "BIAS":
                    P["b"][L["out"].index(t[2])] = float(t[3])
                elif kind == "ZONE":
                    o, i = L["out"].index(t[2]), L["in"].index(t[3])
                    P["W"][o, i] += np.outer(band(t[4], H), band(t[5], W)) * float(t[6])
                elif kind == "LINK":
                    P["W"][L["out"].index(t[2]), L["in"].index(t[3])] = float(t[4])
                else:
                    raise ValueError(f"unknown edit kind {kind}")
            except (ValueError, IndexError) as e:
                raise ValueError(f"edit line {n}: {raw!r}: {e}") from None
        return self

    # ---- forward -----------------------------------------------------------
    def forward(self, x):
        """x: 28x28 array in [0, 1]. Returns the trace dict of every named value."""
        T = {"x": np.asarray(x, dtype=float)}
        self._run(self.arch["layers"], T)
        return T

    def _run(self, layers, T):
        for L in layers:
            getattr(self, "_" + L["op"])(L, T)

    def _loop(self, L, T):
        for _ in range(L["n"]):
            self._run(L["body"], T)

    def _conv(self, L, T):
        P, k = self.P[L["id"]], L["k"]
        c = k // 2
        X = np.stack([np.pad(T[n], c) for n in L["in"]])
        out = np.zeros((len(L["out"]), H, W)) + P["b"][:, None, None]
        for o, i, dy, dx in zip(*np.nonzero(P["W"])):
            out[o] += P["W"][o, i, dy, dx] * X[i, dy:dy + H, dx:dx + W]
        out = ACT[L["act"]](out)
        for o, n in enumerate(L["out"]):
            T[n] = out[o]

    def _ray(self, L, T):
        """Exclusive running max along a direction: is there ink anywhere to the left/right/up/down?"""
        m = T[L["in"]]
        for name, d in L["out"].items():
            if d == "left":
                r = np.maximum.accumulate(m, axis=1)
                r = np.pad(r, ((0, 0), (1, 0)))[:, :W]
            elif d == "right":
                r = np.maximum.accumulate(m[:, ::-1], axis=1)[:, ::-1]
                r = np.pad(r, ((0, 0), (0, 1)))[:, 1:]
            elif d == "up":
                r = np.maximum.accumulate(m, axis=0)
                r = np.pad(r, ((1, 0), (0, 0)))[:H]
            elif d == "down":
                r = np.maximum.accumulate(m[::-1], axis=0)[::-1]
                r = np.pad(r, ((0, 1), (0, 0)))[1:]
            T[name] = r

    def _linesum(self, L, T):
        """Every pixel gets the sum of its whole line: a scan-line count. Lines: "row", "col",
        "d1" (the / diagonal through the pixel, y + x constant), "d2" (the \\ diagonal, y - x constant)."""
        m = T[L["in"]]
        d = L["dir"]
        if d == "row":
            T[L["out"]] = np.repeat(m.sum(1, keepdims=True), W, 1)
        elif d == "col":
            T[L["out"]] = np.repeat(m.sum(0, keepdims=True), H, 0)
        else:
            y, x = np.mgrid[:H, :W]
            key = (y + x) if d == "d1" else (y - x + W - 1)
            sums = np.bincount(key.ravel(), weights=m.ravel(), minlength=H + W - 1)
            T[L["out"]] = sums[key]

    def _plusmax(self, L, T):
        """Max over the pixel and its 4 neighbours (one step of a 4-connected flood)."""
        p = np.pad(T[L["in"]], 1)
        T[L["out"]] = np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])

    def _max3(self, L, T):
        """Max over the 3x3 neighbourhood (thickens strokes by 1 px; closes 1-2 px gaps)."""
        p = np.pad(T[L["in"]], 1)
        T[L["out"]] = np.maximum.reduce([p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])

    def _zone(self, L, T):
        P = self.P[L["id"]]
        X = np.stack([T[n] for n in L["in"]])
        v = np.einsum("fihw,ihw->f", P["W"], X) + P["b"]
        for o, n in enumerate(L["out"]):
            T[n] = float(ACT[L["act"]](v[o]))

    def _linear(self, L, T):
        P = self.P[L["id"]]
        v = P["W"] @ np.array([T[n] for n in L["in"]], dtype=float) + P["b"]
        v = ACT[L["act"]](v)
        for o, n in enumerate(L["out"]):
            T[n] = float(v[o])

    def _maxout(self, L, T):
        """Each class takes the score of its best-matching handwriting style."""
        for out, ins in L["groups"].items():
            T[out] = max(T[n] for n in ins)

    # ---- outputs -----------------------------------------------------------
    def classify(self, x):
        T = self.forward(x)
        classes = self.arch["classes"]
        scores = np.array([T["cls:" + c] for c in classes])
        return classes[int(np.argmax(scores))], scores, T

    def counts(self):
        total = sum(p.size for d in self.P.values() for p in d.values())
        nonzero = sum(int(np.count_nonzero(p)) for d in self.P.values() for p in d.values())
        return total, nonzero


def load(arch_path, edits_path):
    with open(arch_path) as f:
        arch = json.load(f)
    with open(edits_path) as f:
        return Net(arch).apply_edits(f.read())
