"""Robustness checks, run the same way for each model (base file + three seed files):
1. two-level bootstrap: resample training seeds AND questions, so the interval includes seed-to-seed variation;
2. label-free letter-habit correction on KazMMLU STEM (divide each letter's probability by that run's average
   probability for the letter, among questions with the same number of options; the answer key is not used);
3. the held-out questions split by similarity to the nearest training question (>= 0.6 = 'near').

Usage:  python robustness_checks.py heldout_similarity.csv
"""
import sys

import numpy as np
import pandas as pd

KEY = ["set", "subset", "item_id"]
MODELS = {"Llama-3.1-8B-Instruct": ("results_base_llama.csv", [f"results_tuned_llama_seed{s}.csv" for s in (1, 2, 3)]),
          "Qwen2.5-7B-Instruct": ("results_base_qwen.csv", [f"results_tuned_qwen_seed{s}.csv" for s in (1, 2, 3)])}


def load(path):
    return pd.read_csv(path, dtype={"item_id": str})


def corrected_accuracy(r):
    k = r[r.set == "kazmmlu_stem"]
    right = []
    for n, g in k.groupby("n_options"):
        letters = list("ABCDE")[:n]
        p = g[[f"p_{L}" for L in letters]].to_numpy()
        pred = np.array(letters)[(p / p.mean(0)).argmax(1)]
        right.append(pred == g.gold.to_numpy())
    return np.concatenate(right).mean()


def two_level(c0, seeds, rng, B=10000):
    """c0: (n,) before; seeds: (S, n) after. Resample seeds, then questions."""
    S, n = seeds.shape
    out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, S, S)
        q = rng.integers(0, n, n)
        out[b] = seeds[s][:, q].mean() - c0[q].mean()
    return np.percentile(out, [2.5, 97.5])


def main():
    sim = pd.read_csv(sys.argv[1]).set_index("id")
    rng = np.random.default_rng(2026)
    rows = []
    for model, (base_path, seed_paths) in MODELS.items():
        base = load(base_path)
        runs = [load(p) for p in seed_paths]
        m = base[KEY + ["correct"]].rename(columns={"correct": "c0"})
        for k, r in enumerate(runs, 1):
            m = m.merge(r[KEY + ["correct"]].rename(columns={"correct": f"s{k}"}), on=KEY)
        S = ["s1", "s2", "s3"]
        print(f"\n===== {model} =====")
        for name, g in [("own_test", m[m.set == "own_test"]), ("kazmmlu_stem", m[m.set == "kazmmlu_stem"]),
                        ("kazmmlu_nonstem", m[m.set == "kazmmlu_nonstem"])]:
            lo, hi = two_level(g.c0.to_numpy(), g[S].to_numpy().T, rng)
            ch = g[S].to_numpy().mean() - g.c0.mean()
            print(f"two-level 95% interval  {name:16s} change {ch * 100:+5.1f}  [{lo * 100:+5.1f}, {hi * 100:+5.1f}]")
            rows.append(dict(model=model, check="two_level", test=name, change=ch, ci_low=lo, ci_high=hi))

        c_before = corrected_accuracy(base)
        c_after = np.mean([corrected_accuracy(r) for r in runs])
        raw_before = base[base.set == "kazmmlu_stem"].correct.mean()
        raw_after = np.mean([r[r.set == "kazmmlu_stem"].correct.mean() for r in runs])
        print(f"KazMMLU STEM raw        {raw_before:.1%} -> {raw_after:.1%} ({(raw_after - raw_before) * 100:+.1f})")
        print(f"KazMMLU STEM corrected  {c_before:.1%} -> {c_after:.1%} ({(c_after - c_before) * 100:+.1f})  (letter habit removed)")
        rows.append(dict(model=model, check="letter_corrected", test="kazmmlu_stem", before=c_before, after=c_after,
                         change=c_after - c_before))

        own = m[m.set == "own_test"].copy()
        own["near"] = own.item_id.map(sim["near"])
        assert own.near.notna().all()
        for near, g in own.groupby("near"):
            d = g[S].mean(axis=1) - g.c0
            label = "near (>= 0.6)" if near else "not near"
            print(f"held-out {label:14s} n={len(g):3d}  {g.c0.mean():.1%} -> {g[S].to_numpy().mean():.1%}  change {d.mean() * 100:+.1f}")
            rows.append(dict(model=model, check="heldout_similarity", test=label, n=len(g), before=g.c0.mean(),
                             after=g[S].to_numpy().mean(), change=d.mean()))
    pd.DataFrame(rows).to_csv("robustness_checks.csv", index=False)


if __name__ == "__main__":
    main()
