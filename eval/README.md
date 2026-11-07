# eval/

The tools used to study handwriting and score WP-CharNet while its weights were written by hand (see `../LOG.md`).
None of them changes a weight: they read images, run the network and print what they find.

## Data

Datasets are not included. Download them into `data/` at the repository root (it is in `.gitignore`):

| Dataset | Source | Goes to |
|---|---|---|
| MNIST | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/); the same files are mirrored at `https://ossci-datasets.s3.amazonaws.com/mnist/` | `data/mnist/*.gz` |
| EMNIST | [NIST: The EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), "gzip" release: [gzip.zip](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip) (about 560 MB) | `data/emnist/gzip/*.gz` |

```bash
mkdir -p data/mnist data/emnist/gzip
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -L -o data/mnist/$f.gz https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
done
curl -L -o data/emnist/gzip.zip https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
# training files only: test files stay inside the zip until a final test
unzip -j data/emnist/gzip.zip "gzip/emnist-*-train-*" "gzip/emnist-*-mapping.txt" -d data/emnist/gzip/
# for a final test run only (mnist.py --emnist reads the Digits test; there is no Letters test runner yet):
# unzip -j data/emnist/gzip.zip "gzip/emnist-digits-test-*" -d data/emnist/gzip/
```

The study tools read **training** splits only. Test splits are read by `mnist.py` alone, which is meant to be run
once on finished weights. Scores and features are cached in `results/` (not published: some caches hold EMNIST
images) and rebuilt when the edit list changes.

Run everything from the repository root, for example `.venv/bin/python eval/plan.py Dd`.

## Scoring

| Script | What it does |
|---|---|
| `mnist.py <arch> <edits> <out.json> [--emnist]` | test split of MNIST or EMNIST Digits; writes totals and the confusion matrix only |
| `round.py [--holdout]` | one line of scores after a change: MNIST digits and EMNIST letters, own kind and all 62 classes, look-alikes merged; `--holdout` uses the images set aside |
| `byclass.py table [study\|holdout]` | every letter, capital and small apart, on balanced EMNIST ByClass sets |
| `reliability.py` | how often an answer shown at a given percentage was right (held-out images); writes `web/model/reliability.json` for the demo |
| `devset.py` | the drawn test characters of `gen/handwriting.py` |

## Writing a character's rules

The per-character method of `LOG.md`:

| Script | What it does |
|---|---|
| `byclass.py one <c> [sheet.png]` | what class `c` is read as, a sheet of misread images with the wrong answer under each, and the terms that cost it |
| `plan.py <classes>` | for each class it is mistaken for, the features that separate them (means: misread / correctly read / the other class) |
| `split.py <c>` | its own best style against the winner's, and the referees that push it down |
| `missing.py <c>` | which expected parts of its style are absent on misread images |
| `byclass.py contrast <c> <p>` | features separating `c` read as `p` from real `p` |
| `watch.py <classes>` | regenerate the weights, then print the mean and the listed classes |
| `letters.py score\|sheet\|errors\|contrast\|cross` | the same kind of study on EMNIST Letters (cases merged), and digit / letter contrasts |

## Digits (MNIST training split)

| Script | What it does |
|---|---|
| `study.py sheet\|errors\|confusion` | image sheets and confusion on MNIST training images (a held-out chunk is never shown) |
| `contrast.py a b [--err]` | features separating two digits |
| `referees.py` | for misread digits: was the pair's referee right but unheard, or wrong |

## Explaining one answer

| Script | What it does |
|---|---|
| `explain.py <char>` | on the drawn test characters: the top styles and the terms behind them |
| `trace.py <index> [style ...] [--maps out.png]` | one MNIST training image: top classes, key features, chosen styles, feature maps |
