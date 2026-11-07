/* WP-CharNet engine in JavaScript: a line-for-line port of charnet/engine.py.
 *
 * It reads the same two files as the Python engine: the architecture (arch-*.json) and the
 * plain-text edit list (charnet-*.edits). Every parameter starts at zero and only the edits
 * write into it; nothing is converted or precomputed. Runs in the browser and in Node.
 *
 * Also here: the input pipeline the demo uses (strokes -> picture -> 28x28 the MNIST way),
 * and the hand-chosen temperature that turns class scores into the probabilities shown.
 */
(function (root) {
  "use strict";
  const H = 28, W = 28, N = H * W;

  // ---- parameters from the edit list -------------------------------------------------------
  function band(spec, n) {
    let ramp = 0;
    if (spec.includes("~")) { const p = spec.split("~"); spec = p[0]; ramp = parseFloat(p[1]); }
    const [a, b] = spec.split(":").map(parseFloat);
    const m = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const c = i + 0.5;
      if (ramp > 0) {
        const up = Math.min(1, Math.max(0, (c - (a - ramp)) / ramp));
        const dn = Math.min(1, Math.max(0, ((b + ramp) - c) / ramp));
        m[i] = Math.min(1, up, dn);
      } else m[i] = (c >= a && c < b) ? 1 : 0;
    }
    return m;
  }

  const ACT = {
    relu: v => (v > 0 ? v : 0),
    clip01: v => (v < 0 ? 0 : v > 1 ? 1 : v),
    none: v => v,
  };

  function* allLayers(layers) {
    for (const L of layers) {
      if (L.op === "loop") yield* allLayers(L.body);
      else yield L;
    }
  }

  class Net {
    constructor(arch) {
      this.arch = arch;
      this.L = {};
      this.P = {};
      for (const L of allLayers(arch.layers)) {
        this.L[L.id] = L;
        if (L.op === "conv" || L.op === "zone" || L.op === "linear") {
          L._o = indexOf(L.out); L._i = indexOf(L.in);
          // conv: per output, Map "i,dy,dx" -> weight;  zone: per output, Map i -> mask;  linear: Map i -> weight
          this.P[L.id] = { W: L.out.map(() => new Map()), b: new Float64Array(L.out.length) };
        }
      }
    }

    applyEdits(text) {
      const lines = text.split("\n");
      this.why = {};   // "layer out" -> the author's comment on that neuron's BIAS line
      for (let n = 0; n < lines.length; n++) {
        const line = lines[n].split("#")[0].trim();
        if (!line) continue;
        const t = line.split(/\s+/), kind = t[0], L = this.L[t[1]], P = this.P[t[1]];
        const need = (map, k) => { const v = map[k]; if (v === undefined) throw new Error(`edit line ${n + 1}: unknown name ${k}`); return v; };
        if (kind === "TAP") {
          const o = need(L._o, t[2]), i = need(L._i, t[3]);
          P.W[o].set(`${i},${+t[4]},${+t[5]}`, parseFloat(t[6]));          // assignment, as in Python
        } else if (kind === "BIAS") {
          P.b[need(L._o, t[2])] = parseFloat(t[3]);
          const c = lines[n].indexOf("#");
          if (c >= 0) this.why[t[1] + " " + t[2]] = lines[n].slice(c + 1).trim();
        } else if (kind === "ZONE") {
          const o = need(L._o, t[2]), i = need(L._i, t[3]);
          const r = band(t[4], H), c = band(t[5], W), s = parseFloat(t[6]);
          let m = P.W[o].get(i);
          if (!m) { m = new Float64Array(N); P.W[o].set(i, m); }
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) m[y * W + x] += r[y] * c[x] * s;   // ZONE adds
        } else if (kind === "LINK") {
          P.W[need(L._o, t[2])].set(need(L._i, t[3]), parseFloat(t[4]));  // assignment
        } else throw new Error(`edit line ${n + 1}: unknown edit kind ${kind}`);
      }
      // Compile conv taps into flat lists (drop zeros, like np.nonzero in the Python engine).
      for (const L of allLayers(this.arch.layers)) {
        if (L.op !== "conv") continue;
        const P = this.P[L.id];
        P.taps = P.W.map(m => [...m].filter(([, v]) => v !== 0).map(([k, v]) => { const [i, dy, dx] = k.split(",").map(Number); return [i, dy, dx, v]; }));
      }
      return this;
    }

    // ---- forward ----------------------------------------------------------------------------
    forward(x) {
      const T = { x: Float64Array.from(x) };
      this._run(this.arch.layers, T);
      return T;
    }
    _run(layers, T) { for (const L of layers) this["_" + L.op](L, T); }
    _loop(L, T) { for (let k = 0; k < L.n; k++) this._run(L.body, T); }

    _conv(L, T) {
      const P = this.P[L.id], act = ACT[L.act], X = L.in.map(n => T[n]);
      L.out.forEach((name, o) => {
        const out = new Float64Array(N).fill(P.b[o]);
        for (const [i, dy, dx, v] of P.taps[o]) {
          const src = X[i];
          const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
          for (let y = y0; y < y1; y++) {
            const ro = y * W, ri = (y + dy) * W + dx;
            for (let xx = x0; xx < x1; xx++) out[ro + xx] += v * src[ri + xx];
          }
        }
        for (let p = 0; p < N; p++) out[p] = act(out[p]);
        T[name] = out;
      });
    }

    _ray(L, T) {   // exclusive running max: is there ink anywhere to the left/right/up/down?
      const m = T[L.in];
      for (const [name, d] of Object.entries(L.out)) {
        const r = new Float64Array(N);
        if (d === "left" || d === "right") {
          for (let y = 0; y < H; y++) {
            let acc = -Infinity;
            for (let k = 0; k < W; k++) {
              const x = d === "left" ? k : W - 1 - k;
              r[y * W + x] = k === 0 ? 0 : acc;
              acc = Math.max(acc, m[y * W + x]);
            }
          }
        } else {
          for (let x = 0; x < W; x++) {
            let acc = -Infinity;
            for (let k = 0; k < H; k++) {
              const y = d === "up" ? k : H - 1 - k;
              r[y * W + x] = k === 0 ? 0 : acc;
              acc = Math.max(acc, m[y * W + x]);
            }
          }
        }
        T[name] = r;
      }
    }

    _linesum(L, T) {   // every pixel gets the sum of its line: "row", "col", "d1" (/ : y + x const), "d2" (\\ : y - x const)
      const m = T[L.in], r = new Float64Array(N);
      const key = L.dir === "row" ? (y, x) => y : L.dir === "col" ? (y, x) => x : L.dir === "d1" ? (y, x) => y + x : (y, x) => y - x + W - 1;
      const sums = new Float64Array(H + W);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) sums[key(y, x)] += m[y * W + x];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) r[y * W + x] = sums[key(y, x)];
      T[L.out] = r;
    }

    _nbmax(m, offs) {   // max over listed neighbours; outside the picture counts as 0
      const r = new Float64Array(N);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let v = -Infinity;
        for (const [dy, dx] of offs) {
          const yy = y + dy, xx = x + dx;
          const u = (yy < 0 || yy >= H || xx < 0 || xx >= W) ? 0 : m[yy * W + xx];
          if (u > v) v = u;
        }
        r[y * W + x] = v;
      }
      return r;
    }
    _plusmax(L, T) { T[L.out] = this._nbmax(T[L.in], [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]); }
    _max3(L, T) { T[L.out] = this._nbmax(T[L.in], [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]); }

    _zone(L, T) {
      const P = this.P[L.id], act = ACT[L.act], X = L.in.map(n => T[n]);
      L.out.forEach((name, o) => {
        let v = P.b[o];
        for (const [i, m] of P.W[o]) { const src = X[i]; for (let p = 0; p < N; p++) v += m[p] * src[p]; }
        T[name] = act(v);
      });
    }

    _linear(L, T) {
      const P = this.P[L.id], act = ACT[L.act], X = L.in.map(n => T[n]);
      L.out.forEach((name, o) => {
        let v = P.b[o];
        for (const [i, w] of P.W[o]) v += w * X[i];
        T[name] = act(v);
      });
    }

    _maxout(L, T) {   // each class takes the score of its best-matching handwriting style
      for (const [out, ins] of Object.entries(L.groups)) T[out] = Math.max(...ins.map(n => T[n]));
    }

    scores(T) { return this.arch.classes.map(c => T["cls:" + c]); }

    // The terms behind one neuron of a linear layer: [input name, weight, value], largest effect first.
    terms(layerId, outName, T) {
      const L = this.L[layerId], P = this.P[layerId], o = L._o[outName];
      return [...P.W[o]].filter(([, w]) => w !== 0).map(([i, w]) => [L.in[i], w, T[L.in[i]]])
        .sort((a, b) => Math.abs(b[1] * b[2]) - Math.abs(a[1] * a[2]));
    }
  }

  function indexOf(list) { const m = {}; list.forEach((n, i) => { m[n] = i; }); return m; }

  // ---- probabilities ---------------------------------------------------------------------------
  // Hand-chosen, not fitted. Class scores are sums of evidence; the referee stage already treats a
  // class 1.5 below the top as out of the running (REF_GATE). At TEMPERATURE = 0.5 such a class gets
  // e^-3, about 1/20 of the top class's probability: "out of the running" reads as 20-to-1 against.
  const TEMPERATURE = 0.5;
  function probabilities(scores, temp = TEMPERATURE) {
    const top = Math.max(...scores);
    const e = scores.map(s => Math.exp((s - top) / temp));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / z);
  }

  // ---- input pipeline ---------------------------------------------------------------------------
  // Lanczos-3 resampling (the filter PIL's LANCZOS uses), separable, float.
  function lanczos(t) {
    if (t === 0) return 1;
    if (t <= -3 || t >= 3) return 0;
    const a = Math.PI * t;
    return 3 * Math.sin(a) * Math.sin(a / 3) / (a * a);
  }
  function resize1d(src, sw, sh, dw, axis) {   // resample along x (axis 0) or y (axis 1)
    const inN = axis === 0 ? sw : sh, outN = dw, other = axis === 0 ? sh : sw;
    const scale = inN / outN, fs = Math.max(scale, 1), support = 3 * fs;
    const out = new Float64Array(axis === 0 ? outN * sh : sw * outN);
    for (let i = 0; i < outN; i++) {
      const c = (i + 0.5) * scale;
      const lo = Math.max(0, Math.floor(c - support)), hi = Math.min(inN, Math.ceil(c + support));
      const w = []; let tot = 0;
      for (let j = lo; j < hi; j++) { const v = lanczos((j - c + 0.5) / fs); w.push(v); tot += v; }
      for (let k = 0; k < other; k++) {
        let s = 0;
        for (let j = lo; j < hi; j++) s += w[j - lo] * (axis === 0 ? src[k * sw + j] : src[j * sw + k]);
        s /= tot || 1;
        if (axis === 0) out[k * outN + i] = s; else out[i * sw + k] = s;
      }
    }
    return out;
  }

  // Same steps as gen/handwriting.py normalize(img, "mnist"): bounding box into 20x20 keeping the
  // aspect ratio, then the centre of mass moved to the centre of the 28x28 picture.
  function pyround(v) {   // Python's round(): halves go to the even neighbour
    const f = Math.floor(v), d = v - f;
    if (Math.abs(d - 0.5) < 1e-12) return f % 2 === 0 ? f : f + 1;
    return Math.round(v);
  }

  function normalize(img, w, h) {
    let y0 = h, y1 = -1, x0 = w, x1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (img[y * w + x] > 0.05) {
      if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x;
    }
    const out = new Float64Array(N);
    if (y1 < 0) return out;
    const ch = y1 - y0 + 1, cw = x1 - x0 + 1, crop = new Float64Array(ch * cw);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = img[(y + y0) * w + x + x0];
    const f = 20 / Math.max(ch, cw);
    const nh = Math.max(1, pyround(ch * f)), nw = Math.max(1, pyround(cw * f));
    const small = resize1d(resize1d(crop, cw, ch, nw, 0), nw, ch, nh, 1);
    for (let p = 0; p < small.length; p++) small[p] = Math.min(1, Math.max(0, Math.round(small[p] * 255) / 255));
    let tot = 0, sy = 0, sx = 0;
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) { const v = small[y * nw + x]; tot += v; sy += v * y; sx += v * x; }
    tot = tot || 1;
    let oy = pyround(13.5 - sy / tot), ox = pyround(13.5 - sx / tot);
    oy = Math.min(Math.max(oy, 0), 28 - nh); ox = Math.min(Math.max(ox, 0), 28 - nw);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) out[(y + oy) * W + x + ox] = small[y * nw + x];
    return out;
  }

  // Draw pen strokes the way gen/handwriting.py render() does: a 112 px square, the character
  // scaled into 70% of it, a round pen whose width is a fixed share of the character's height
  // (MNIST strokes are about a tenth of the height; the dev set used 7-17%). So it does not
  // matter how big or small someone writes on the pad. strokes: [[[x, y], ...], ...]
  const PEN = 0.12, SIZE = 112;
  function rasterize(strokes) {
    const pts = strokes.flat();
    const img = new Float64Array(SIZE * SIZE);
    if (!pts.length) return img;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    const scale = SIZE * 0.7 / Math.max(x1 - x0, y1 - y0, 1e-6);
    const ox = SIZE / 2 - (x0 + x1) / 2 * scale, oy = SIZE / 2 - (y0 + y1) / 2 * scale;
    const r = Math.max(PEN * (y1 - y0) * scale, 0.07 * SIZE * 0.7) / 2;   // a flat stroke (a dash) still gets a pen
    for (const st of strokes) {
      const q = st.map(([x, y]) => [x * scale + ox, y * scale + oy]);
      const segs = q.length > 1 ? q.slice(1).map((p, i) => [q[i], p]) : [[q[0], q[0]]];   // a tap is a dot
      for (const [[ax, ay], [bx, by]] of segs) {
        const lo = v => Math.max(0, Math.floor(v - r - 1)), hiX = v => Math.min(SIZE - 1, Math.ceil(v + r + 1));
        const dx = bx - ax, dy = by - ay, dd = dx * dx + dy * dy;
        for (let y = lo(Math.min(ay, by)); y <= hiX(Math.max(ay, by)); y++)
          for (let x = lo(Math.min(ax, bx)); x <= hiX(Math.max(ax, bx)); x++) {
            const px = x + 0.5, py = y + 0.5;
            const t = dd ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / dd)) : 0;
            const d = Math.hypot(px - ax - t * dx, py - ay - t * dy);
            const v = Math.min(1, Math.max(0, r - d + 0.5));                // 1 px anti-aliased edge
            if (v > img[y * SIZE + x]) img[y * SIZE + x] = v;
          }
      }
    }
    return img;
  }
  function fromStrokes(strokes) { return normalize(rasterize(strokes), SIZE, SIZE); }

  const api = { Net, normalize, rasterize, fromStrokes, probabilities, TEMPERATURE, H, W };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CharNet = api;
})(typeof self !== "undefined" ? self : this);
