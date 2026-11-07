"""Score WP-CharNet on the author-drawn development set (clean drawings with writer variation)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter, defaultdict
from charnet.engine import load
from gen.handwriting import devset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Characters that are the same shape once rescaled to the box (no size or baseline cue left).
SAME = [set("0Oo"), set("1Il"), set("cC"), set("sS"), set("vV"), set("wW"), set("xX"), set("zZ"),
        set("pP"), set("kK"), set("uU"), set("yY"), set("jJ")]


def same_shape(a, b):
    return a == b or any(a in g and b in g for g in SAME)


def main(n=6, seed=1, show=25):
    net = load(os.path.join(ROOT, "arch.json"), os.path.join(ROOT, "weights", "charnet.edits"))
    ok = ok_case = tot = 0
    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    for img, ch, style, mode in devset(n, seed):
        pred, _, _ = net.classify(img)
        tot += 1
        hit = pred == ch
        ok += hit
        ok_case += same_shape(ch, pred)
        per[f"{ch}/{style}"][0] += hit
        per[f"{ch}/{style}"][1] += 1
        if not hit:
            conf[(ch, pred)] += 1
    print(f"dev exact {ok}/{tot} = {ok / tot:.1%}   merging same-shape characters {ok_case / tot:.1%}")
    worst = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:show]
    print("weakest styles:", "  ".join(f"{k} {a}/{b}" for k, (a, b) in worst))
    print("top confusions (true->pred):", "  ".join(f"{a}->{b} {c}" for (a, b), c in conf.most_common(show)))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
