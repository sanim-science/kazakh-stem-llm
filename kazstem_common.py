"""Pure-Python helpers shared by the evaluation and training notebooks.

Everything here is copied verbatim into the notebooks (one cell), so the prompt used
for training is exactly the prompt used for evaluation. No torch needed.
"""
from __future__ import annotations

import math

import pandas as pd

LETTERS = "ABCDE"
STEM_SUBSETS = ["Math (High School in kaz)", "Physics (High School in kaz)", "Chemistry (High School in kaz)",
                "Biology (High School in kaz)", "Informatics (High School in kaz)"]
NONSTEM_SUBSETS = ["Kazakh History (High School in kaz)", "Geography (High School in kaz)"]
SUBJECT_EN = {  # subset or own-subject -> words used in the prompt
    "Math (High School in kaz)": "mathematics", "Physics (High School in kaz)": "physics",
    "Chemistry (High School in kaz)": "chemistry", "Biology (High School in kaz)": "biology",
    "Informatics (High School in kaz)": "informatics", "Kazakh History (High School in kaz)": "history of Kazakhstan",
    "Geography (High School in kaz)": "geography",
    "math": "mathematics", "physics": "physics", "chemistry": "chemistry", "biology": "biology",
    "informatics": "informatics",
}


def build_messages(subject_en: str, question: str, options: dict) -> list:
    """The one prompt used everywhere: English instruction, Kazakh question and options."""
    letters = [L for L in LETTERS if str(options.get(L, "")).strip()]
    lines = [f"The following is a multiple choice question about {subject_en} (high school level, in Kazakh).",
             f"Reply with only the letter of the correct answer ({', '.join(letters)}).", "",
             str(question).strip(), ""]
    lines += [f"{L}. {str(options[L]).strip()}" for L in letters]
    lines += ["", "Answer:"]
    return [{"role": "user", "content": "\n".join(lines)}]


def items_from_kazmmlu(df: pd.DataFrame, subset: str, set_name: str):
    """Rows of a KazMMLU CSV -> list of item dicts. Read the CSV with dtype=str, keep_default_na=False
    so that options like 'NA' (sodium) or 'None' stay text."""
    items, skipped = [], 0
    for _, r in df.iterrows():
        opts = {L: str(r.get(f"Option {L}", "")).strip() for L in LETTERS}
        gold = str(r.get("Answer Key", "")).strip().upper()
        avail = [L for L in LETTERS if opts[L]]
        if gold not in avail or len(avail) < 2 or not str(r.get("Question", "")).strip():
            skipped += 1
            continue
        items.append(dict(set=set_name, subset=subset, item_id=str(r.get("ID", "")), gold=gold,
                          messages=build_messages(SUBJECT_EN[subset], r["Question"], opts), letters="".join(avail)))
    return items, skipped


def items_from_own(df: pd.DataFrame, set_name: str = "own_test"):
    """test_own.csv from validate_questions.py: id, subject, topic, question, A..E, answer."""
    items, skipped = [], 0
    for _, r in df.iterrows():
        opts = {L: str(r.get(L, "")).strip() for L in LETTERS}
        gold = str(r.get("answer", "")).strip().upper()
        avail = [L for L in LETTERS if opts[L]]
        if gold not in avail:
            skipped += 1
            continue
        items.append(dict(set=set_name, subset=str(r["subject"]), item_id=str(r["id"]), gold=gold,
                          messages=build_messages(SUBJECT_EN[str(r["subject"])], r["question"], opts),
                          letters="".join(avail)))
    return items, skipped


def pick_letter(letter_logprobs: dict, letters: str):
    """letter -> log-probability; only letters that exist in this question compete.
    Returns (prediction, probabilities renormalised over the available letters)."""
    avail = [L for L in letters]
    m = max(letter_logprobs[L] for L in avail)
    ex = {L: math.exp(letter_logprobs[L] - m) for L in avail}
    z = sum(ex.values())
    probs = {L: ex[L] / z for L in avail}
    return max(avail, key=lambda L: probs[L]), probs


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(res: pd.DataFrame) -> dict:
    """Accuracy per set and per subset with 95 % Wilson intervals, plus chance level."""
    out = {}
    for s, g in res.groupby("set"):
        k, n = int(g["correct"].sum()), len(g)
        per = {}
        for sub, h in g.groupby("subset"):
            kk, nn = int(h["correct"].sum()), len(h)
            per[sub] = dict(n=nn, accuracy=kk / nn, ci95=wilson(kk, nn), chance=float((1 / h["n_options"]).mean()))
        out[s] = dict(n=n, accuracy=k / n, ci95=wilson(k, n), chance=float((1 / g["n_options"]).mean()),
                      macro_accuracy=float(sum(v["accuracy"] for v in per.values()) / len(per)), by_subset=per)
    return out
