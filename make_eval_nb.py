"""Build eval_kazmmlu.ipynb. The helper cell is copied from kazstem_common.py so the notebook
and the local tests use the same code."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
common = open(os.path.join(HERE, "kazstem_common.py"), encoding="utf-8").read()
common = re.sub(r'^""".*?"""\n', "", common, flags=re.S).replace("from __future__ import annotations\n", "")
common = "# Shared helpers: the same prompt is used for evaluation and training.\n" + common.strip() + "\n"

MD_INTRO = """# KazMMLU Kazakh STEM: evaluation

Scores one model on
* the five Kazakh STEM subjects of [KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU) (test split, about 4,200 questions),
* two Kazakh non-STEM subjects (Kazakh History, Geography), to check that fine-tuning doesn't damage other knowledge,
* your own 413 held-out questions (`test_own.csv` from the kazakh-stem-2000 dataset).

**Method.** Zero-shot. Each question is shown once with an English instruction; the model's probability for each answer letter is read off, and the most likely letter that exists in that question is its answer (the method KazMMLU used for open models). Same code before and after fine-tuning: only `USE_ADAPTER` changes.

**How to run.** Right panel: Accelerator = GPU T4 x2, Internet = On, Add-ons → Secrets → HF_TOKEN switched on (needed for Llama). Add Input → Your Datasets → kazakh-stem-2000. Edit the settings cell, then Save Version → Save & Run All.

**Output** (Output tab): `results_<RUN_NAME>.csv` (one row per question) and `summary_<RUN_NAME>.json`. Send both to Claude.
"""

SETTINGS = """# ===== Settings: change only these =====
MODEL = "llama"          # "llama" = meta-llama/Llama-3.1-8B-Instruct (needs approved Meta access)
                         # "qwen"  = Qwen/Qwen2.5-7B-Instruct (no access form)
USE_ADAPTER = False      # False = starting score (Step 2); True = score with your trained adapter (Step 5)
RUN_NAME = "base"        # names the output files, e.g. "base", "tuned_seed1"
BATCH_SIZE = 8           # lower to 4 if you see "CUDA out of memory"
INCLUDE_NONSTEM = True   # also score two non-STEM Kazakh subjects
"""

SETUP = """import glob, json, math, os, subprocess, sys, time
if USE_ADAPTER:      # Kaggle ships torchao 0.10, which peft refuses to load next to; this project does not use torchao
    import importlib.util
    if importlib.util.find_spec("torchao") is not None:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
        print("removed torchao (not used here; its old version blocks peft)")
import pandas as pd
import torch, transformers
from packaging import version

print("torch", torch.__version__, "| transformers", transformers.__version__)
n_gpu = torch.cuda.device_count()
mem = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n_gpu)]
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n_gpu)], [f"{m:.0f} GB" for m in mem])
if n_gpu == 0 or sum(mem) < 28:
    raise RuntimeError("Needs GPU T4 x2 (two 15 GB GPUs): right panel -> Session options -> Accelerator.")
if version.parse(transformers.__version__) < version.parse("4.45"):
    raise RuntimeError("transformers is older than 4.45; send this message to Claude.")

HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN secret found")
except Exception as e:
    print("No HF_TOKEN secret (fine for Qwen, needed for Llama):", type(e).__name__)

if USE_ADAPTER:
    try:
        import peft
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
        import peft
    print("peft", peft.__version__)
"""

DATA = """from huggingface_hub import hf_hub_download, list_repo_files

REPO = "MBZUAI/KazMMLU"
files = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN)

def test_file(subset):
    inside = [f for f in files if f.startswith(subset + "/") and f.lower().endswith(".csv")]
    tests = [f for f in inside if "test" in f.rsplit("/", 1)[-1].lower()]
    if len(tests) != 1:
        raise RuntimeError(f"{subset}: expected one test CSV, found {inside}")
    return tests[0]

items, skipped = [], {}
for sub in STEM_SUBSETS + (NONSTEM_SUBSETS if INCLUDE_NONSTEM else []):
    path = hf_hub_download(REPO, test_file(sub), repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)      # keep 'NA', 'None' as text
    got, sk = items_from_kazmmlu(df, sub, "kazmmlu_stem" if sub in STEM_SUBSETS else "kazmmlu_nonstem")
    items += got
    skipped[sub] = sk
    print(f"{sub:40s} {len(got):5d} questions ({sk} skipped: no valid answer key)")

own = sorted(glob.glob("/kaggle/input/**/test_own.csv", recursive=True))
if own:
    df = pd.read_csv(own[0], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    got, sk = items_from_own(df)
    items += got
    skipped["own_test"] = sk
    print(f"{'your held-out questions':40s} {len(got):5d} questions ({sk} skipped)  [{own[0]}]")
else:
    print("test_own.csv not found: Add Input -> Your Datasets -> kazakh-stem-2000. Continuing with KazMMLU only.")
print("total:", len(items))
"""

MODEL_CELL = """from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen2.5-7B-Instruct"}
model_id = MODEL_IDS[MODEL]
try:
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
except Exception as e:
    if MODEL == "llama":
        raise RuntimeError("Can't open Llama: Meta hasn't approved your access yet, or the HF_TOKEN secret is off. "
                           "Set MODEL = 'qwen' in the settings cell, or wait for Meta's email.") from e
    raise
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

t_load = time.time()
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)
adapter_dir = None
if USE_ADAPTER:
    from peft import PeftModel
    cfgs = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    if len(cfgs) != 1:
        raise RuntimeError(f"Expected one adapter in the inputs (adapter_config.json), found {cfgs}. Add your adapter dataset as an input.")
    adapter_dir = os.path.dirname(cfgs[0])
    print("adapter:", adapter_dir, "| trained on:", json.load(open(cfgs[0])).get("base_model_name_or_path"))
    model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()
print(f"loaded {model_id} in {time.time() - t_load:.0f} s")
"""

SCORING = """def letter_ids(tok):
    ids = {}
    for L in LETTERS:
        c = set()
        for s in (L, " " + L):                       # 'A' and ' A' can be different tokens
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                c.add(t[0])
        if not c:
            raise RuntimeError(f"no single-token form of {L}")
        ids[L] = sorted(c)
    return ids

LIDS = letter_ids(tok)
print("letter token ids:", LIDS)
for it in items:                                     # the chat template is applied once per question
    it["text"] = tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def score(batch):
    enc = tok([it["text"] for it in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
    out = []
    for i, it in enumerate(batch):
        llp = {L: torch.logsumexp(lp[i, LIDS[L]], dim=0).item() for L in LETTERS}
        pred, probs = pick_letter(llp, it["letters"])
        out.append(dict(pred=pred, probs=probs, letter_mass=sum(math.exp(llp[L]) for L in it["letters"])))
    return out
"""

SMOKE = """# Quick check on 2 questions per subject before the full run
probe = []
for sub in dict.fromkeys(it["subset"] for it in items):
    probe += [it for it in items if it["subset"] == sub][:2]
print("----- what the model sees (first question) -----")
print(probe[0]["text"])
print("------------------------------------------------")
probe_res = score(probe)
for it, r in zip(probe, probe_res):
    print(f"{it['subset'][:32]:32s} gold {it['gold']}  pred {r['pred']}  p={r['probs'][r['pred']]:.2f}  "
          f"probability on answer letters {r['letter_mass']:.2f}")
mass = sum(r["letter_mass"] for r in probe_res) / len(probe)
print(f"mean probability on answer letters: {mass:.2f}")
if mass < 0.5:
    print("WARNING: the model puts little probability on the answer letters. Send this output to Claude before trusting the scores.")
"""

RUN = """order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))   # similar lengths together = less padding
rows = [None] * len(items)
t0 = time.time()
for b in range(0, len(order), BATCH_SIZE):
    idx = order[b:b + BATCH_SIZE]
    for i, r in zip(idx, score([items[i] for i in idx])):
        it = items[i]
        rows[i] = dict(set=it["set"], subset=it["subset"], item_id=it["item_id"], gold=it["gold"], pred=r["pred"],
                       correct=int(r["pred"] == it["gold"]), n_options=len(it["letters"]),
                       letter_mass=round(r["letter_mass"], 4),
                       **{f"p_{L}": round(r["probs"].get(L, 0.0), 4) for L in LETTERS})
    if (b // BATCH_SIZE) % 50 == 0:
        done, el = b + len(idx), time.time() - t0
        print(f"{done}/{len(items)} questions, {el / 60:.1f} min, about {el / done * (len(items) - done) / 60:.0f} min left", flush=True)
res = pd.DataFrame(rows)
res.to_csv(f"/kaggle/working/results_{RUN_NAME}.csv", index=False)
print(f"done in {(time.time() - t0) / 60:.1f} min")
"""

SUMMARY = """summary = dict(run=RUN_NAME, model=model_id, adapter=adapter_dir,
               method="zero-shot, English instruction, chat template, answer-letter log-probability",
               dtype="float16", transformers=transformers.__version__, torch=torch.__version__,
               minutes=round((time.time() - t0) / 60, 1), skipped=skipped,
               mean_letter_mass=round(float(res["letter_mass"].mean()), 4), results=summarize(res))
json.dump(summary, open(f"/kaggle/working/summary_{RUN_NAME}.json", "w"), indent=1, ensure_ascii=False)
for s, v in summary["results"].items():
    lo, hi = v["ci95"]
    print(f"\\n{s}: {v['accuracy']:.1%} (95% CI {lo:.1%} to {hi:.1%}), n = {v['n']}, guessing would give {v['chance']:.0%}")
    for sub, w in v["by_subset"].items():
        print(f"    {sub:40s} {w['accuracy']:.1%}   n = {w['n']}")
print(f"\\nmean probability on answer letters: {summary['mean_letter_mass']:.2f}")
print("Download results_*.csv and summary_*.json from the Output tab and send them to Claude.")
"""


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"cells": [cell("markdown", MD_INTRO), cell("code", SETTINGS), cell("code", SETUP), cell("code", common),
                cell("code", DATA), cell("code", MODEL_CELL), cell("code", SCORING), cell("code", SMOKE),
                cell("code", RUN), cell("code", SUMMARY)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
if __name__ == "__main__":
    out = os.path.join(HERE, "eval_kazmmlu.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
