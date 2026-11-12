"""Three-seed result: per question, average the fine-tuned correctness over seeds and compare with the
starting model on the same question. 95 % interval and two-sided p from 10,000 bootstrap resamples of
questions; subject-level p-values also Holm-adjusted within each test set (5 STEM subjects, 5 own).

Usage:  python aggregate.py [--out=file.csv] results_base_llama.csv results_tuned_llama_seed1.csv seed2.csv seed3.csv
Writes  three_seed_summary.csv (or --out)
"""
import sys

import numpy as np
import pandas as pd

KEY = ["set", "subset", "item_id"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    args = sys.argv[1:]
    out_path = "three_seed_summary.csv"
    if args and args[0].startswith("--out="):
        out_path = args.pop(0).split("=", 1)[1]
    base = pd.read_csv(args[0], dtype={"item_id": str})[KEY + ["correct"]].rename(columns={"correct": "c0"})
    seeds = []
    for k, path in enumerate(args[1:], 1):
        t = pd.read_csv(path, dtype={"item_id": str})[KEY + ["correct"]].rename(columns={"correct": f"s{k}"})
        base = base.merge(t, on=KEY)
        seeds.append(f"s{k}")
    assert len(base) == 6632, len(base)
    base["tuned"] = base[seeds].mean(1)
    rng = np.random.default_rng(2026)
    rows = []

    def row(name, level, g):
        d = (g["tuned"] - g["c0"]).to_numpy()
        boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        per_seed = [g[s].mean() - g["c0"].mean() for s in seeds]
        rows.append(dict(test=name, level=level, n=len(g), before=g["c0"].mean(), after_mean=g["tuned"].mean(),
                         change=d.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5), p=p,
                         seed_changes=" / ".join(f"{x * 100:+.1f}" for x in per_seed)))

    for s, g in base.groupby("set"):
        row(s, "set", g)
        start = len(rows)
        for sub, h in g.groupby("subset"):
            row(f"{s} / {sub}", "subject", h)
        adj = holm([r["p"] for r in rows[start:]])
        for r, a in zip(rows[start:], adj):
            r["p_holm"] = a
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"{'test':52s} {'n':>5s} {'before':>7s} {'after':>7s} {'change':>7s} {'95% CI':>16s} {'p':>7s} {'p_holm':>7s}  per seed")
    for r in out.itertuples():
        ph = "" if pd.isna(getattr(r, "p_holm", np.nan)) else f"{r.p_holm:.2g}"
        print(f"{r.test[:52]:52s} {r.n:5d} {r.before:7.1%} {r.after_mean:7.1%} {r.change * 100:+6.1f}  "
              f"[{r.ci_low * 100:+5.1f}, {r.ci_high * 100:+5.1f}] {r.p:7.2g} {ph:>7s}  {r.seed_changes}")


if __name__ == "__main__":
    main()
