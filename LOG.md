# How the weights were written

WP-CharNet's weights were written by hand in a generator (`gen/generate.py`, `gen/rules.py`): every value is typed
with a reason, nothing is fitted. This log records how the network was built, the evidence behind each decision,
and the attempts that did not work.

## Data and its use

- **Studied:** MNIST training images and EMNIST training images (Digits, Letters, ByClass). Claude viewed
  sheets of images (correctly read and misread), traces of single images, and per-class feature averages.
  Nothing was computed from them; they informed hand-written rules.
- **Held out:** MNIST train[6000:9000]; EMNIST Letters train[60000:65200]; 120 images per letter class from EMNIST
  ByClass train[400000:440000]. Never viewed; only totals were checked, once per round of changes.
- **Study sets for letters:** the first 300 images of each of the 52 letter classes in EMNIST ByClass train[0:120000]
  (`eval/byclass.py`), so capitals and small letters are scored separately.
- **Test splits:** the EMNIST Letters and ByClass test splits have never been opened. The MNIST and EMNIST Digits
  test splits were each run once on an earlier stage of these weights (aggregate scores only), so they are no
  longer perfectly clean for them.
- **Drawn test characters** (`gen/handwriting.py`): clean characters written as pen strokes, with writer variation, used as a quick check that
  drawn input (as in the demo) behaves like scanned handwriting.

Letters are scored case-folded ("a" for an A counts as right): after scaling to the box, many capitals and small
letters have the same shape.

## The method, per character

1. **What it is read as**, and a sheet of its misread images with the wrong answer under each (`byclass.py one`).
2. **What separates it** from each class it is mistaken for: every feature ranked by separation, with three means,
   misread / correctly read / the other class (`eval/plan.py`). Only features where the correctly read images
   agree with the misread ones are used; a feature the real character also has would push it down.
3. **Where its score goes** (`eval/split.py`): its own best style against the winner's, and which referees push it.
4. **Which expected parts are missing** on misread images (`eval/missing.py`).
5. Then: rewrite the rule, add a referee, or add a neuron; test the character, the classes it competes with, and
   the mean (`eval/watch.py`, `eval/round.py`).

## Decisions that shaped the network

**Digits first.** Digit rules and 29 digit referees were written from MNIST training images: scan-line counts
(how many strokes a row or column crosses), a branch counter (a 16-point ring of radius 5 counts the strokes leaving
each point: junction = 3, crossing = 4), and referees for confused pairs such as 3/2, 4/9, 0/8, 5/6, 7/1.

**Referees push only the loser.** When a referee's winner also gained points, a digit that won several pairs
jumped over letters no referee had compared it with: on letter images the best digit gained 1.18 on average, and a
letter beat every digit 5% of the time (43% before the referee stage). Now a referee only pushes the loser of its
pair down.

**Only the strongest loss counts.** On misread a's the a rule scored above the winner (0.37 vs 0.32), but eight
referees, each saying "not a, a little", together pushed it down by 1.67. A class now drops by its strongest single
referee loss (`say:x>y` neurons, `loss:c` maxout). Mean over letters 61.5% → 61.9%; g 36% → 46%; Q 59% → 64%.

**Separate outputs for one kind.** `dig:*` and `let:*` rank digits only or letters only and hear only referees
within that kind; a digit pushed down by a letter referee had been falling below another digit (digits-only lost
2.7 points before this).

**Position-free neurons.** Narrow characters are centred in the frame, so "a stem in the left third" missed most
b's (fired on 3% of misread b's). New neurons see relations instead: `stem_lh` / `stem_rh` (a vertical stroke with a
hole to its right / left on the same row), `asc_l` / `asc_r` / `desc_l` / `desc_r` (the stem continuing above or
below its bowl, on hole-free rows: b, d, p, q). A bowl's own sides also count as "a stem beside a hole", so b and d
are told apart by the ascender. Rules for H, U, u, R, D that used frame-based stems (present on 1–5% of misread
images) were rewritten with row counts (`hx2@U`, `hx2@D`).

**Diagonal stroke counts.** Strokes crossed by each / and \\ line (`ax1-3`, `bx1-3`), as rows and columns already
had: these separate slanted letters (A, V, K, X, M, N, W, Z) and S (whose spine runs down to the right and is
crossed twice by the / lines).

**Whole-letter neurons.** An AND of a letter's parts, for the letters where parts alone matched too much:
`a_body` (bowl left of a straight stem ending in a tail at the bottom right), `B_body`, `b_body`, `d_body`,
`D_body`, `E_arms` (two openings to the right around a middle arm), `e_body`, `S_body`, `tail_below_r` /
`tail_below_l` (a bowl with one stroke hanging below it: q, g, p).

**Referees whose evidence was part of the true letter** were the most common cause of a weak letter. Examples:
A/R used R's own hole and legs as A evidence (R went 20% → 72% when rewritten on "an R opens to the right between
bowl and leg; an A's crossbar meets its right leg"); J/T used the opening left of a T's stem (T 37% → 83%); c/S
counted ink in the middle left, which an S's upper curve also has. Each was found with `split.py` and rewritten on
features the true character does not have, or removed.

**Rules that expected perfect handwriting** were relaxed where the penalty fired on the real letter: an S's upper
curve is a left edge at the top; a thick N's diagonal has top edges; a D's overlapping stroke makes a small second
hole (a "waist"); a handwritten e often ends at the top; a K's meeting point is a crossing (four branches).

**Every rule written once.** The rules file lists one entry per referee pair; a test (`tests/test_rules.py`) fails if
any style, concept or referee key is written twice (a Python dict literal would silently keep only the last).

## What did not work

- Lowering every penalty (unexpected parts 0.3 → 0.15, contradictions 0.5 → 0.35): letters 54.0% → 52.8%. Penalties
  also keep wrong classes out; lowering them everywhere lifts rivals as much as the true class.
- Straightening the slant before reading (moment deskew): letters 54.0% → 49.9%. A, V, W, M, N have lopsided ink, so
  the shear tilts them, and 1 and 7 lose their slant cues.
- Requiring a stem for K hurt real K's until the "crossing" penalty was removed (a K's meeting point is a crossing).
- A D/O referee "a D's stem fills its left corners": O fell 63% → 18% (thick O's fill them too).
- A D "overshoot" style and position referees for D/P and D/p: P 89% → 70–75%.
- E and e rules rewritten around new neurons cut E to 20% and e to 24%; the neurons stayed as optional parts.
- Referees against g/q: no feature separates EMNIST g from q by more than 0.2; they stay a look-alike pair with 9.

## Where it stands

Held-out training images (letters-only view, case-folded; digits-only for MNIST):

| | Held out |
|---|---|
| EMNIST ByClass letters, mean of 52 classes | 72.5% |
| EMNIST Letters | 73.1% (all 62: 61.3%) |
| MNIST digits | 89.3% (all 62: 61.8%) |

Per class on the held-out ByClass set:

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A 87 | a 68 | B 78 | b 77 | C 77 | c 87 | D 59 | d 89 | E 70 | e 78 | F 68 | f 60 | G 64 | g 52 |
| H 88 | h 81 | I 38 | i 45 | J 71 | j 57 | K 82 | k 68 | L 56 | l 46 | M 83 | m 87 | N 57 | n 73 |
| O 73 | o 71 | P 92 | p 92 | Q 72 | q 44 | R 62 | r 69 | S 68 | s 57 | T 89 | t 59 | U 75 | u 79 |
| V 78 | v 76 | W 88 | w 89 | X 89 | x 84 | Y 78 | y 74 | Z 91 | z 78 | | | | |

Size: 11,679 edit lines, 188,521 nonzero parameters (23.3 M including zeros), 94 styles, 16 concepts, 307 referees.

## Known limits and next steps

- I, i, l, L as plain sticks look like 1; only a serif, a dot or a foot separates them.
- g and q are drawn alike in EMNIST.
- D's stem often overshoots into a P or b shape.
- Next neurons to try: a slant-tolerant ascender (b and d that lean far right light both ascender neurons), a hook
  detector (g, j, q tails), a "dot above" check for i and j.
