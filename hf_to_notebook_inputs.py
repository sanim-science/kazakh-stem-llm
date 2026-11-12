"""Turn the Hugging Face dataset into the two files the Kaggle notebooks read:
questions_clean.jsonl (all 2,000 questions with their split) and test_own.csv (the 413 held-out questions).

Usage:
    python hf_to_notebook_inputs.py                 # downloads Sanim09/kazakh-stem-mcq-2000 with `datasets`
    python hf_to_notebook_inputs.py train.jsonl test.jsonl   # or from the two data files
Then upload both files as one Kaggle dataset and add it as an input to the notebooks.
"""
import json
import sys

import pandas as pd

LETTERS = "ABCDE"


def load():
    if len(sys.argv) == 3:
        return {s: pd.read_json(p, lines=True, dtype={"id": str}) for s, p in zip(("train", "test"), sys.argv[1:])}
    from datasets import load_dataset
    ds = load_dataset("Sanim09/kazakh-stem-mcq-2000")
    return {s: ds[s].to_pandas() for s in ("train", "test")}


def main():
    parts = load()
    with open("questions_clean.jsonl", "w", encoding="utf-8") as f:
        for split, df in parts.items():
            for r in df.itertuples():
                rec = dict(id=r.id, subject=r.subject, grade=str(r.grade), question_kk=r.question,
                           **{f"option_{L}": getattr(r, L) for L in LETTERS}, answer=r.answer, split=split)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    t = parts["test"]
    out = pd.DataFrame({"id": t.id, "subject": t.subject, "topic": "", "question": t.question,
                        **{L: t[L] for L in LETTERS}, "answer": t.answer})
    out.to_csv("test_own.csv", index=False, encoding="utf-8-sig")
    print(f"wrote questions_clean.jsonl ({sum(len(d) for d in parts.values())} questions) and test_own.csv ({len(out)})")


if __name__ == "__main__":
    main()
