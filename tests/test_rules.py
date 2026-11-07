"""Checks on the hand-written rules themselves (gen/rules.py)."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dict_keys(name):
    tree = ast.parse(open(os.path.join(ROOT, "gen", "rules.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            return [k.value for k in node.value.keys]
    raise AssertionError(f"{name} not found")


def test_no_repeated_keys():
    """A Python dict literal keeps only the last of two equal keys, silently; every rule must be written once."""
    for table in ("STYLES", "CONCEPTS", "REFEREES"):
        keys = _dict_keys(table)
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        assert not repeated, f"{table}: written twice: {repeated}"


def test_one_referee_per_pair():
    pairs = [frozenset(k.split(">")) for k in _dict_keys("REFEREES")]
    assert len(pairs) == len(set(pairs)), "a pair of classes has two referees"
