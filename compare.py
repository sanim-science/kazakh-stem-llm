"""Paired comparison of two evaluation runs (same questions): accuracy change, exact McNemar test,
95 % bootstrap interval for the change, per subject, and the answer-letter habits of each run.

Usage:  python compare.py results_base_llama.csv results_tuned_llama_seed1.csv [more tuned runs ...]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

KEY = ["set", "subset", "item_id"]


def load(path):
    r = pd.read_csv(path, dtype={"item_id": str})
    assert not r.duplicated(KEY).any(), f"duplicate questions in {path}"
    return r


def paired(a, b, rng, B=10000):
    """a, b: 0/1 arrays for the same questions. Returns change, CI, McNemar p, b->right, b->wrong."""
    up = int(((a == 0) & (b == 1)).sum())
    down = int(((a == 1) & (b == 0)).sum())
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    idx = rng.integers(0, len(a), (B, len(a)))
    d = (b[idx].mean(1) - a[idx].mean(1))
    return b.mean() - a.mean(), np.percentile(d, [2.5, 97.5]), p, up, down


def main():
    base = load(sys.argv[1])
    rng = np.random.default_rng(2026)
    for path in sys.argv[2:]:
        tuned = load(path)
        m = base.merge(tuned, on=KEY, suffixes=("_base", "_tuned"))
        assert len(m) == len(base) == len(tuned), "runs cover different questions"
        assert (m.gold_base == m.gold_tuned).all(), "answer keys differ"
        print(f"\n=== {path} vs {sys.argv[1]}  ({len(m)} questions) ===")
        print(f"{'test':42s} {'before':>7s} {'after':>7s} {'change':>7s}  {'95% CI':>15s}  {'wrong->right':>12s} {'right->wrong':>12s}  McNemar p")
        groups = [(s, g) for s, g in m.groupby("set")] + [(f"  {s} / {sub}", h) for (s, sub), h in m.groupby(["set", "subset"])]
        groups.sort(key=lambda x: x[0].strip())
        for name, g in groups:
            a, b = g.correct_base.to_numpy(), g.correct_tuned.to_numpy()
            d, ci, p, up, down = paired(a, b, rng)
            print(f"{name[:42]:42s} {a.mean():7.1%} {b.mean():7.1%} {d * 100:+6.1f}  [{ci[0] * 100:+5.1f}, {ci[1] * 100:+5.1f}]  "
                  f"{up:12d} {down:12d}  {p:.2g}")
        print("\nanswer-letter habits on 4-option KazMMLU STEM questions (share of answers):")
        k4 = m[(m.set == "kazmmlu_stem") & (m.n_options_base == 4)]
        for col, label in (("gold_base", "correct key"), ("pred_base", "before"), ("pred_tuned", "after")):
            v = k4[col].value_counts(normalize=True).reindex(list("ABCD"), fill_value=0)
            print(f"   {label:12s} " + "  ".join(f"{L} {v[L]:.0%}" for L in "ABCD"))


if __name__ == "__main__":
    main()
