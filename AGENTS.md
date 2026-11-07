# AGENTS.md

Guidance for AI agents (and people) improving WP-CharNet. Read `README.md` for what the model is and `LOG.md` for how
its weights were written; this file is about how to work on it.

## What this project is

A 62-class handwritten-character classifier (0–9, A–Z, a–z) whose weights are **written by hand**. The generator
(`gen/generate.py`, `gen/rules.py`) emits a plain-text edit list (`weights/charnet.edits`); the engine
(`charnet/engine.py`, mirrored in `web/charnet.js`) starts every parameter at zero and applies it. Every map,
feature and neuron has a name; every value has a written reason. The point is a network a reader can follow neuron
by neuron, in the spirit of WPLM (linked in `README.md`).

## House rules

1. **No training, ever.** No gradient descent, no fitting, no automatic search over weights or thresholds, no
   templates or stored examples, no statistics copied from data into weights. You may *look at* training images and
   feature averages, then type values by hand. A sweep of a constant is allowed only as a few hand-picked settings,
   each logged.
2. **Every value has a reason.** Rules, concepts and referees carry a `why` string that describes exactly what their
   terms test (positive terms are evidence for the first class of a referee, negative for the second). If you
   change terms, rewrite the reason.
3. **Training splits only.** Study and tune on training data. Test splits stay closed; they are run once on finished
   weights, and the result is reported whatever it is. The EMNIST Letters and ByClass test splits have never been
   opened; MNIST and EMNIST Digits test were each used once on an earlier stage of these weights (see `LOG.md`).
4. **Held-out images are for totals only.** Never view them; check them once per round of changes:
   MNIST train[6000:9000], EMNIST Letters train[60000:65200], ByClass `holdout` set (`eval/byclass.py`).
5. **Log everything** in `LOG.md`: what you looked at, what you changed, the numbers before and after, and
   especially what did not work.
6. **Do not break other classes.** After any change, check the class you worked on, the classes it competes with,
   the study mean (`eval/watch.py`), and the digits (`eval/round.py`). A gain that only moves images from one class to
   its look-alike is not a gain.
7. **One set of weights.** No version names in files or docs; the repository holds the current network only.
8. **Keep the browser engine identical.** If you change `charnet/engine.py` (a new op or behaviour), change
   `web/charnet.js` the same way and check that both give the same scores.
9. **Privacy.** No personal names, emails or local paths in files or commits. Datasets and `results/` caches are
   never committed (some caches hold EMNIST images).
10. **No assistant-specific files.** This file is the shared guide; do not add `CLAUDE.md` or other agent files
    (they are in `.gitignore`).

## How the network is built (short)

- **Feature maps (convolutions):** ink, stroke pieces and stroke ends (8 directions), rays, scan-line stroke counts
  along rows / columns / diagonals, a branch counter (junction = 3 strokes meet, crossing = 4), holes and 3-sided
  openings (flood fill, 24 steps), and position-free relations to holes (`stem_lh/rh`, `asc_*`, `desc_*`).
- **L7 zone features:** how much of each map lies in each zone, e.g. `bay_R@T` (opening to the right, top band).
  Zones: `A` all, `T/M/B` rows, `l/c/r` columns, `Tl`… both; scan maps also `U/D` (upper/lower half).
- **L8 concepts** (`CONCEPTS` in `rules.py`): small ANDs/ORs of L7 features, including whole-letter neurons
  (`a_body`, `B_body`, `E_arms`, `S_body` …). Concepts read L7 features only, not other concepts.
- **L9 styles** (`STYLES`): one neuron per way of writing a character:
  `"x/name": "expected*w ?optional | contradiction*w"`. Positive weights are normalised to sum to 1; contradictions
  subtract 0.5×w; rules are closed-world (a salient part the style does not mention costs 0.3). A class takes its
  best style (L10).
- **Referees** (`REFEREES`, L11–L13M): one per look-alike pair, `"a>b": (terms, bias, why)`, heard only while both
  classes are within 1.5 of the top. A referee only pushes its loser down, and a class drops by its **strongest**
  single loss, never the sum.
- **Outputs (L14):** `cls:*` (all 62), `dig:*` (digits only, digit referees only), `let:*` (letters only).

## Decisions to keep (and why)

- Referees never lift winners (lifting let a digit that won several pairs jump over letters).
- Only the strongest referee loss counts (summed losses buried letters that look like many others, e.g. a).
- Prefer **position-free** features (relations, row/column counts) over frame zones like "stem in the left third":
  narrow characters are centred in the frame, so frame-based stems fire on only a few percent of real ones.
- Letters are scored case-folded; many capitals and small letters share a shape once scaled to the box.
- Probabilities use a hand-picked softmax temperature (0.5). Do not fit it; the demo shows measured reliability
  instead.

## How to improve a character

1. `byclass.py one <c> sheet.png`: look at what it is read as and at the misread images.
2. `split.py <c>`: is its own rule weak, or is a referee pushing it down? The most common cause of a weak letter
   was a referee whose evidence for the other class is **also part of the true letter**.
3. `plan.py <c>`: for each thief, features with three means (misread / correctly read / thief). Use only features
   where the correctly read images agree with the misread ones.
4. `missing.py <c>`: which expected parts of its style are absent (a part absent on most real letters is either
   detected badly or not how people write it).
5. Change one thing: a rule's terms, a new style (a class takes its best style, so adding a style is safer than
   rewriting one), a referee, a concept, or a new feature map.
6. `watch.py <c><thieves>`: test the class and its neighbours; revert what does not help; log the result.

## Gotchas

- **Never write a key twice** in `STYLES`, `CONCEPTS` or `REFEREES`: a Python dict literal silently keeps only the
  last one. `tests/test_rules.py` catches this; run the tests.
- In a term list, a repeated feature keeps its **last** weight (the generator's `LINK` is an assignment).
- Concepts may only use L7 feature names; the generator checks every name before writing.
- `multiprocessing` workers on macOS re-import the main script: run tools from files, not from stdin.
- Caches in `results/` are keyed by the edit list's modification time; regenerate before scoring.
- After changing weights, copy `arch.json` and `weights/charnet.edits` into `web/model/` (the GitHub Pages
  workflow does this on deploy) and re-run `eval/reliability.py` for the demo's table.
- On a case-insensitive disk, `e.png` and `E.png` are the same file; name sheets per case.

## Commands

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy pillow scipy pytest pyyaml
.venv/bin/python gen/generate.py          # write arch.json and weights/charnet.edits
.venv/bin/python -m pytest -q tests       # per-neuron tests and rule checks
.venv/bin/python eval/watch.py aAgq       # regenerate and score the listed classes (study set)
.venv/bin/python eval/round.py            # digits and letters together (add --holdout once per round)
cd web && python3 -m http.server          # the demo
```

Datasets: see `eval/README.md`.

## Where to go next

- A slant-tolerant ascender (b and d that lean far right light both ascender neurons).
- A hook detector for the tails of g, j, q, and a "dot above" check for i and j.
- D: its stem often overshoots into a P or b shape.
- With all 62 classes, about a third of digit images still go to a look-alike letter.
- Weakest classes on held-out images: I, i, l, q, L, g, j, D, t (38–59%).

## Definition of done for a change

Tests pass; the class improved on the study set without its neighbours or the digits dropping; the held-out check
agrees; reasons match terms; `LOG.md` has an entry; `web/model/` is in sync if the weights changed.
