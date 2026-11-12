"""Build overlap_check.ipynb: compares your 2,000 questions with KazMMLU's Kazakh STEM questions,
by wording (character n-grams + same numbers) and by meaning (LaBSE sentence embeddings)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

MD = """# Overlap check: your questions vs KazMMLU

Before training, every one of your 2,000 questions is compared with every KazMMLU Kazakh STEM question (test and dev splits):

* **Wording:** character n-gram similarity, plus a rule for the same numbers reworded.
* **Meaning:** [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) multilingual sentence embeddings, which also catch paraphrases.

A pair is flagged for review when the meaning score is 0.90 or more, or the wording check finds a near-copy or the same numbers reworded. Flagged is not the same as contaminated: you and Claude review each pair and drop only real duplicates.

**How to run.** Right panel: Internet = On, Accelerator = GPU T4 x2 (CPU also works, slower). Add Input → Your Datasets → kazakh-stem-2000. Save Version → Save & Run All.

**Output:** `overlap_flagged.csv` (pairs to review) and `overlap_top1.csv` (closest KazMMLU question for each of yours). Send both to Claude.
"""

SETTINGS = """# ===== Settings =====
LABSE_FLAG = 0.90      # meaning similarity at or above this is flagged
LABSE_SHOW = 0.85      # pairs between SHOW and FLAG are listed as 'borderline'
"""

SETUP = """import glob, json, re, subprocess, sys
import numpy as np, pandas as pd
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import hf_hub_download, list_repo_files
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

NUM = re.compile(r"\\d+(?:[.,]\\d+)?")          # ordinary digits; superscripts and subscripts are not counted

def numsig(s):
    \"\"\"Sorted numbers in a question if it has at least two: the same data reworded keeps them.\"\"\"
    n = sorted(x.replace(",", ".") for x in NUM.findall(str(s)))
    return tuple(n) if len(n) >= 2 else None
"""

DATA = """ours = [json.loads(l) for l in open(glob.glob("/kaggle/input/**/questions_clean.jsonl", recursive=True)[0], encoding="utf-8")]
ours = pd.DataFrame(ours)
print("your questions:", len(ours), ours["split"].value_counts().to_dict())

REPO = "MBZUAI/KazMMLU"
STEM = ["Math (High School in kaz)", "Physics (High School in kaz)", "Chemistry (High School in kaz)",
        "Biology (High School in kaz)", "Informatics (High School in kaz)"]
files = list_repo_files(REPO, repo_type="dataset")
ref = []
for sub in STEM:
    for f in [f for f in files if f.startswith(sub + "/") and f.lower().endswith(".csv")]:
        d = pd.read_csv(hf_hub_download(REPO, f, repo_type="dataset"), dtype=str, keep_default_na=False)
        d["subset"], d["file"] = sub, f.rsplit("/", 1)[-1]
        ref.append(d)
ref = pd.concat(ref, ignore_index=True)
ref = ref[ref["Question"].str.strip() != ""].reset_index(drop=True)
print("KazMMLU questions:", len(ref), ref.groupby(["subset", "file"]).size().to_dict())

def with_options(q, opts):
    return str(q).strip() + " | " + " | ".join(str(o).strip() for o in opts if str(o).strip())

ours_q = ours["question_kk"].tolist()
ref_q = ref["Question"].tolist()
ours_full = [with_options(r.question_kk, [r.option_A, r.option_B, r.option_C, r.option_D, r.option_E]) for r in ours.itertuples()]
ref_full = [with_options(r["Question"], [r.get(f"Option {L}", "") for L in "ABCDE"]) for _, r in ref.iterrows()]
"""

LEXICAL = """vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(ours_q + ref_q)
LEX = (vec.transform(ours_q) @ vec.transform(ref_q).T).toarray()
print("wording similarity computed:", LEX.shape)
"""

SEMANTIC = """model = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)
E_ours = model.encode(ours_full, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
E_ref = model.encode(ref_full, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
SEM = E_ours @ E_ref.T
print("meaning similarity computed:", SEM.shape)
"""

COMBINE = """by_sig = {}
for j, q in enumerate(ref_q):
    if numsig(q):
        by_sig.setdefault(numsig(q), []).append(j)

top, flagged = [], []
for i, r in ours.iterrows():
    js = int(SEM[i].argmax()); jl = int(LEX[i].argmax())
    reasons = []
    if SEM[i, js] >= LABSE_FLAG:
        reasons.append(("meaning", js, f"meaning {SEM[i, js]:.2f}"))
    if LEX[i, jl] >= 0.90 and not (numsig(r.question_kk) and numsig(ref_q[jl]) and numsig(r.question_kk) != numsig(ref_q[jl])):
        reasons.append(("near-copy", jl, f"wording {LEX[i, jl]:.2f}"))
    if numsig(r.question_kk) in by_sig:
        jj = max(by_sig[numsig(r.question_kk)], key=lambda x: LEX[i, x])
        if LEX[i, jj] >= 0.5:
            reasons.append(("same numbers", jj, f"same numbers, wording {LEX[i, jj]:.2f}"))
    top.append(dict(id=r.id, split=r.split, subject=r.subject, meaning_top1=round(float(SEM[i, js]), 4),
                    wording_top1=round(float(LEX[i, jl]), 4), kazmmlu_id_meaning=ref.at[js, "ID"],
                    kazmmlu_subset=ref.at[js, "subset"]))
    level = "flag" if reasons else ("borderline" if SEM[i, js] >= LABSE_SHOW else None)
    if level:
        kind, j, why = reasons[0] if reasons else ("meaning", js, f"meaning {SEM[i, js]:.2f}")
        flagged.append(dict(level=level, reason="; ".join(x[2] for x in reasons) or why, id=r.id, split=r.split,
                            subject=r.subject, meaning=round(float(SEM[i, j]), 4), wording=round(float(LEX[i, j]), 4),
                            your_question=ours_full[i], kazmmlu_subset=ref.at[j, "subset"], kazmmlu_file=ref.at[j, "file"],
                            kazmmlu_id=ref.at[j, "ID"], kazmmlu_question=ref_full[j]))
top = pd.DataFrame(top)
flagged = pd.DataFrame(flagged)
if len(flagged):
    flagged = flagged.sort_values(["level", "meaning"], ascending=[False, False])
top.to_csv("/kaggle/working/overlap_top1.csv", index=False)
flagged.to_csv("/kaggle/working/overlap_flagged.csv", index=False)

print("closest-match meaning score, your questions:")
print(top["meaning_top1"].describe(percentiles=[.5, .9, .99]).round(3).to_string())
n_flag = int((flagged["level"] == "flag").sum()) if len(flagged) else 0
n_border = int((flagged["level"] == "borderline").sum()) if len(flagged) else 0
print(f"\\nflagged for review: {n_flag}   borderline (meaning {LABSE_SHOW}-{LABSE_FLAG}): {n_border}")
if n_flag:
    print(flagged[flagged.level == "flag"].groupby(["split", "subject"]).size().to_string())
    for r in flagged[flagged.level == "flag"].head(5).itertuples():
        print(f"\\n[{r.id}] {r.reason}\\n  yours:   {r.your_question[:160]}\\n  KazMMLU: {r.kazmmlu_question[:160]}")
print("\\nDownload overlap_flagged.csv and overlap_top1.csv from the Output tab and send them to Claude.")
"""


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"cells": [cell("markdown", MD), cell("code", SETTINGS), cell("code", SETUP), cell("code", DATA),
                cell("code", LEXICAL), cell("code", SEMANTIC), cell("code", COMBINE)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = os.path.join(HERE, "overlap_check.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
