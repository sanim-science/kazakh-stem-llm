"""Build train_lora.ipynb: LoRA fine-tuning on the training questions (training only; score the adapter
afterwards with eval_kazmmlu.ipynb, whose scoring cells are imported from make_eval_nb)."""
import json
import os

from make_eval_nb import DATA, SCORING, SETUP, cell, common

HERE = os.path.dirname(os.path.abspath(__file__))

MD = """# LoRA fine-tuning on Kazakh STEM MCQ 2000

One run = one seed, **training only**:
* A LoRA adapter is trained on the training questions (the 413 held-out questions and the 10 flagged by the overlap check are never used). The model stays in 16-bit across both T4s, the same precision as the evaluation; only the adapter (about 0.5% of the weights) learns.
* The model learns to answer each question with its letter, using **exactly the evaluation prompt**. 10% of the training questions (the same ones for every run) are held back; after each epoch the adapter is kept if validation accuracy is the best so far.

**Then evaluate in a fresh session** with eval_kazmmlu.ipynb (`USE_ADAPTER = True`, this notebook's output added as an input). Scoring in the same session right after an hour of training crashed the GPU once; a fresh session is what worked for the starting scores.

**How to run.** Right panel: Accelerator = GPU T4 x2, Internet = On, HF_TOKEN secret on (Llama). Add Input → kazakh-stem-2000. Set `SEED` in the first cell, then Save Version → Save & Run All.

**Output:** the `adapter/` folder, `train_log_<RUN_NAME>.csv` and `train_summary_<RUN_NAME>.json`.

Settings are fixed in advance and are the same for every seed. KazMMLU is only used to score the finished model, never to choose settings.
"""

SETTINGS = """# ===== Settings =====
MODEL = "llama"          # "llama" = meta-llama/Llama-3.1-8B-Instruct | "qwen" = Qwen/Qwen2.5-7B-Instruct
SEED = 1                 # run once each with 1, 2 and 3

# Fixed in advance; keep these the same for every run
EXCLUDE_IDS = ["BIO-0230", "PHY-0329", "MAT-0030", "PHY-0413", "MAT-0357",      # flagged by the
               "MAT-0012", "CHE-0034", "PHY-0360", "PHY-0367", "MAT-0073"]      # overlap check
EPOCHS = 2
LR = 2e-4
LORA_R, LORA_ALPHA = 16, 16
MICRO_BATCH, ACCUM = 4, 2   # 8 questions per update
VAL_FRAC = 0.10
BATCH_SIZE = 8              # questions per batch when scoring
INCLUDE_NONSTEM = True
USE_ADAPTER = True          # makes sure peft is installed
RUN_NAME = f"tuned_{MODEL}_seed{SEED}"
print("run:", RUN_NAME)
"""

TRAIN_DATA = """import hashlib, random
from collections import Counter
import numpy as np

qc = sorted(glob.glob("/kaggle/input/**/questions_clean.jsonl", recursive=True))
if not qc:
    raise RuntimeError("questions_clean.jsonl not found: Add Input -> Your Datasets -> kazakh-stem-2000.")
allq = pd.DataFrame([json.loads(l) for l in open(qc[0], encoding="utf-8")])
missing = sorted(set(EXCLUDE_IDS) - set(allq["id"]))
if missing:
    raise RuntimeError(f"EXCLUDE_IDS not found in the questions: {missing}")
pool = allq[(allq["split"] == "train") & (~allq["id"].isin(EXCLUDE_IDS))].copy()
# the same validation questions for every seed and model: chosen by a hash of the id
pool["h"] = pool["id"].map(lambda x: int(hashlib.sha1(("val-" + x).encode("utf-8")).hexdigest()[:8], 16) / 2**32)

def to_items(df, set_name):
    out = []
    for r in df.itertuples():
        opts = {L: str(getattr(r, f"option_{L}") or "").strip() for L in LETTERS}
        out.append(dict(set=set_name, subset=r.subject, item_id=r.id, gold=r.answer,
                        messages=build_messages(SUBJECT_EN[r.subject], r.question_kk, opts),
                        letters="".join(L for L in LETTERS if opts[L])))
    return out

train_items = to_items(pool[pool["h"] >= VAL_FRAC], "train")
val_items = to_items(pool[pool["h"] < VAL_FRAC], "val")
bad = [it["item_id"] for it in train_items + val_items if it["gold"] not in it["letters"]]
if bad:
    raise RuntimeError(f"answer key points to an empty option: {bad[:10]}")
print(f"training questions: {len(train_items)}   validation: {len(val_items)}   excluded by overlap check: {len(EXCLUDE_IDS)}")
print("training subjects:", dict(Counter(it["subset"] for it in train_items)))
print("training answers: ", dict(sorted(Counter(it["gold"] for it in train_items).items())))
"""

MODEL_CELL = """from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen2.5-7B-Instruct"}
model_id = MODEL_IDS[MODEL]
try:
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
except Exception as e:
    if MODEL == "llama":
        raise RuntimeError("Can't open Llama: switch on the HF_TOKEN secret (Add-ons -> Secrets).") from e
    raise
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

t_load = time.time()
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto", token=HF_TOKEN)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
for p in model.parameters():                 # the adapter trains in float32; the frozen model stays float16
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
model.print_trainable_parameters()
adapter_dir = "/kaggle/working/adapter"
print(f"loaded {model_id} in {time.time() - t_load:.0f} s")
"""

TRAIN = """for it in train_items + val_items:
    it["text"] = tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
TARGET = {}
for L in LETTERS:                            # the reply's first token: the bare letter
    t = tok.encode(L, add_special_tokens=False)
    assert len(t) == 1, L
    TARGET[L] = t[0]

def batch_loss(batch):
    enc = tok([it["text"] for it in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    try:
        out = model(**enc, logits_to_keep=1)
    except TypeError:
        out = model(**enc)
    logits = out.logits[:, -1, :].float()
    y = torch.tensor([TARGET[it["gold"]] for it in batch], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, y)

def val_accuracy():
    model.eval()
    right = 0
    for b in range(0, len(val_items), BATCH_SIZE):
        batch = val_items[b:b + BATCH_SIZE]
        right += sum(r["pred"] == it["gold"] for it, r in zip(batch, score(batch)))
    return right / len(val_items)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
per_epoch = math.ceil(len(train_items) / (MICRO_BATCH * ACCUM))
total = per_epoch * EPOCHS
warm = max(1, round(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
scaler = torch.amp.GradScaler("cuda")

log, best, step, t_train = [], (-1.0, 0), 0, time.time()
acc0 = val_accuracy()
print(f"validation accuracy before training: {acc0:.1%}   ({len(val_items)} questions)")
log.append(dict(epoch=0, step=0, val_accuracy=acc0))
order = random.Random(SEED)
for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = list(range(len(train_items)))
    order.shuffle(idx)
    losses = []
    for b in range(0, len(idx), MICRO_BATCH * ACCUM):
        group = idx[b:b + MICRO_BATCH * ACCUM]
        opt.zero_grad(set_to_none=True)
        for m in range(0, len(group), MICRO_BATCH):
            mb = [train_items[i] for i in group[m:m + MICRO_BATCH]]
            loss = batch_loss(mb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at step {step}")
            scaler.scale(loss * len(mb) / len(group)).backward()
            losses.append(loss.item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 20 == 0 or step == total:
            recent = float(np.mean(losses[-20 * ACCUM:]))
            el = time.time() - t_train
            print(f"epoch {epoch}  step {step}/{total}  loss {recent:.3f}  lr {sched.get_last_lr()[0]:.1e}  "
                  f"{el / 60:.1f} min, about {el / step * (total - step) / 60:.0f} min left", flush=True)
            log.append(dict(epoch=epoch, step=step, train_loss=recent))
    acc = val_accuracy()
    print(f"end of epoch {epoch}: training loss {np.mean(losses):.3f}, validation accuracy {acc:.1%}")
    log.append(dict(epoch=epoch, step=step, train_loss_epoch=float(np.mean(losses)), val_accuracy=acc))
    if acc > best[0]:
        best = (acc, epoch)
        model.save_pretrained(adapter_dir)
        print("  adapter saved (best validation accuracy so far)")
train_minutes = (time.time() - t_train) / 60
pd.DataFrame(log).to_csv(f"/kaggle/working/train_log_{RUN_NAME}.csv", index=False)
print(f"training done in {train_minutes:.1f} min; saved adapter = epoch {best[1]} "
      f"(validation {best[0]:.1%}, before training {acc0:.1%})")
"""

TRAINING_INFO = """info = dict(run=RUN_NAME, model=model_id, seed=SEED, epochs=EPOCHS, best_epoch=best[1],
            val_accuracy_best=best[0], val_accuracy_before=acc0, n_train=len(train_items), n_val=len(val_items),
            excluded=EXCLUDE_IDS, lr=LR, lora_r=LORA_R, lora_alpha=LORA_ALPHA, batch=MICRO_BATCH * ACCUM,
            precision="float16 model, float32 adapter", train_minutes=round(train_minutes, 1))
json.dump(info, open(f"/kaggle/working/train_summary_{RUN_NAME}.json", "w"), indent=1)
print(json.dumps(info, indent=1))
print("\\nNext: score this adapter with eval_kazmmlu.ipynb in a fresh session (USE_ADAPTER = True,")
print(f"RUN_NAME = '{RUN_NAME}', this notebook's output added as an input).")
"""

nb = {"cells": [cell("markdown", MD), cell("code", SETTINGS), cell("code", SETUP), cell("code", common),
                cell("code", DATA), cell("code", TRAIN_DATA), cell("code", MODEL_CELL), cell("code", SCORING),
                cell("code", TRAIN), cell("code", TRAINING_INFO)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    out = os.path.join(HERE, "train_lora.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
