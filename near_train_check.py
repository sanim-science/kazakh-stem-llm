"""Does the KazMMLU change after fine-tuning come from KazMMLU questions that closely resemble a training question?

A KazMMLU question counts as "near a trained question" when it is the closest KazMMLU match (LaBSE meaning
similarity, from overlap_top1.csv) of at least one question the model was actually trained on
(split == train, not in the validation slice, not one of the 10 excluded ids) at similarity >= threshold.
For near vs. other KazMMLU questions: 3-seed mean change, and the difference with a 95 % bootstrap interval.

Usage:  python near_train_check.py [--out=file.csv] overlap_top1.csv results_base_llama.csv seed1.csv seed2.csv seed3.csv
"""
import hashlib
import sys

import numpy as np
import pandas as pd

KEY = ["set", "subset", "item_id"]
EXCLUDE_IDS = ["BIO-0230", "PHY-0329", "MAT-0030", "PHY-0413", "MAT-0357",
               "MAT-0012", "CHE-0034", "PHY-0360", "PHY-0367", "MAT-0073"]
VAL_FRAC = 0.10


def val_hash(x):
    return int(hashlib.sha1(("val-" + x).encode("utf-8")).hexdigest()[:8], 16) / 2**32


def main():
    args = sys.argv[1:]
    out_path = "near_train_check.csv"
    if args and args[0].startswith("--out="):
        out_path = args.pop(0).split("=", 1)[1]
    top = pd.read_csv(args[0])
    pool = top[(top.split == "train") & ~top.id.isin(EXCLUDE_IDS)].copy()
    trained = pool[pool.id.map(val_hash) >= VAL_FRAC]
    print(f"trained questions found in overlap_top1: {len(trained)} (training notebook used 1,408)")

    base = pd.read_csv(args[1], dtype={"item_id": str})
    m = base[KEY + ["correct"]].rename(columns={"correct": "c0"})
    seeds = []
    for k, p in enumerate(args[2:], 1):
        t = pd.read_csv(p, dtype={"item_id": str})[KEY + ["correct"]].rename(columns={"correct": f"s{k}"})
        m = m.merge(t, on=KEY)
        seeds.append(f"s{k}")
    k = m[m.set == "kazmmlu_stem"].copy()
    assert len(k) == 4174, len(k)
    assert not k.item_id.duplicated().any(), "KazMMLU ids repeat across subjects"
    k["d"] = k[seeds].mean(axis=1) - k.c0
    # the overlap check also searched KazMMLU dev.csv; those questions are not in the evaluation, so drop them
    in_eval = trained.kazmmlu_id_meaning.astype(str).isin(set(k.item_id))
    print(f"trained questions whose closest match is a KazMMLU dev question (not evaluated): {(~in_eval).sum()}")
    trained = trained[in_eval]

    rng = np.random.default_rng(2026)

    def ci(x, B=10000):
        b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
        return np.percentile(b, [2.5, 97.5])

    def diff_ci(a, b, B=10000):
        ba = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
        bb = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
        d = ba - bb
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        return np.percentile(d, [2.5, 97.5]), p

    print(f"\n{'scope':12s} {'thr':>5s} {'near n':>6s} {'near change':>12s} {'95% CI':>16s} "
          f"{'other n':>7s} {'other change':>12s} {'difference':>11s} {'95% CI':>16s} {'p':>6s}")
    rows = []
    for thr in (0.70, 0.75, 0.80, 0.85):
        near_ids = set(trained.loc[trained.meaning_top1 >= thr, "kazmmlu_id_meaning"].astype(str))
        k["near"] = k.item_id.isin(near_ids)
        for scope, g in [("all STEM", k)] + [(s.split(" (")[0], h) for s, h in k.groupby("subset")]:
            a, b = g.loc[g.near, "d"].to_numpy(), g.loc[~g.near, "d"].to_numpy()
            if len(a) < 5:
                continue
            (lo, hi), (dlo, dhi), p = ci(a), *diff_ci(a, b)
            rows.append(dict(scope=scope, threshold=thr, near_n=len(a), near_change=a.mean(), near_ci_low=lo,
                             near_ci_high=hi, other_n=len(b), other_change=b.mean(), difference=a.mean() - b.mean(),
                             diff_ci_low=dlo, diff_ci_high=dhi, p=p))
            print(f"{scope:12s} {thr:5.2f} {len(a):6d} {a.mean() * 100:+11.1f}  [{lo * 100:+5.1f}, {hi * 100:+5.1f}] "
                  f"{len(b):7d} {b.mean() * 100:+11.1f}  {(a.mean() - b.mean()) * 100:+10.1f}  "
                  f"[{dlo * 100:+5.1f}, {dhi * 100:+5.1f}] {p:6.2g}")
        print()
    pd.DataFrame(rows).to_csv(out_path, index=False)

    # Chemistry with the near-train questions removed: does the +3.4 survive?
    chem = k[k.subset.str.startswith("Chemistry")]
    for thr in (0.75, 0.80):
        near_ids = set(trained.loc[trained.meaning_top1 >= thr, "kazmmlu_id_meaning"].astype(str))
        rest = chem.loc[~chem.item_id.isin(near_ids), "d"].to_numpy()
        lo, hi = ci(rest)
        print(f"Chemistry without near-train questions (thr {thr:.2f}): n={len(rest)}  "
              f"change {rest.mean() * 100:+.1f}  [{lo * 100:+.1f}, {hi * 100:+.1f}]")


if __name__ == "__main__":
    main()
