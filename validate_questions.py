"""Check the Kazakh STEM question workbook and export it for fine-tuning.

Usage (in the folder with the workbook; save and close it in Excel first):

    pip install pandas openpyxl scikit-learn
    python validate_questions.py kazakh_stem_questions.xlsx
    python validate_questions.py kazakh_stem_questions.xlsx --kazmmlu kazmmlu_math.csv kazmmlu_physics.csv ...
    python validate_questions.py kazakh_stem_questions.xlsx --sheet Examples      (try it on the examples)

What it checks
    ERROR    the row is left out of the export until you fix it
             (missing fields, bad id, answer not A-E, a solution that doesn't end in "Жауабы: X",
              solution letter different from the answer column, duplicate options,
              "A)" typed in an option, exact duplicate question, too close to a KazMMLU question)
    NOT READY  AI draft with verified not "yes" (left out, not an error)
    WARNING  exported, but look at it
             (Latin letters inside Kazakh words, no Kazakh letters at all, Cyrillic А/В/С/Е used as
              the answer letter, only 4 options, numeric options not in increasing order,
              correct option much longer than the others, near-duplicate questions,
              answer letters out of balance, missing date)

What it writes (to export/, or export_<sheet>/ for another sheet)
    report.md               every problem with its Excel row number, plus counts
    questions_clean.jsonl   every exported question with all fields and its split
    sft_train_cot.jsonl     training split, chat format; the reply is the worked solution
                            (only questions that have a solution)
    sft_train_letter.jsonl  training split, chat format; the reply is only the letter
    test_own.csv            held-out split (about 20 %): your own questions the model never trains on
    splits.csv              id -> train/test. Kept between runs, so a question never changes split.
                            Near-duplicate questions always go to the same split.

Exit code 1 if any row has an ERROR.

KazMMLU overlap check: this is a lexical check. It catches copies, near-copies and the same
numbers reworded; it cannot catch a free paraphrase of a word question (e.g. biology). Do a
meaning-based check (LaBSE) on Kaggle before the final run. Download the Kazakh STEM parts of KazMMLU from its Hugging Face page
(CSV, JSONL, Parquet or XLSX) and pass the files with --kazmmlu. The question column is found
automatically; if not, give its name with --kazmmlu-col.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Kazakh text in a Windows console
except Exception:
    pass

SUBJECTS = ["math", "physics", "chemistry", "biology", "informatics"]
PREFIX = {"math": "MAT", "physics": "PHY", "chemistry": "CHE", "biology": "BIO", "informatics": "INF"}
DEFAULT_TARGETS = {"math": 550, "physics": 510, "chemistry": 370, "biology": 370, "informatics": 200}
LETTERS = "ABCDE"
OPTS = [f"option_{L}" for L in LETTERS]
COLS = ["id", "subject", "topic", "grade", "difficulty", "question_kk", *OPTS, "answer", "solution_kk",
        "source", "drafted_by", "verified", "verify_method", "corrections", "date", "notes"]
REQUIRED_OPTS = OPTS[:4]

KAZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYR = "А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі"
TOKEN = re.compile(rf"[A-Za-z{CYR}]+")
HAS_LAT, HAS_CYR = re.compile("[A-Za-z]"), re.compile(f"[{CYR}]")
# Cyrillic look-alikes of the answer letters (А В С Е) and lower case
LOOKALIKE = {"А": "A", "В": "B", "С": "C", "Е": "E", "а": "A", "в": "B", "с": "C", "е": "E",
             "a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
MARK_END = re.compile(r"Жауабы\s*:\s*([A-Ea-eАВСЕавсе])\s*[.!]?\s*$")
MARK_ANY = re.compile(r"Жауабы\s*:\s*([A-Ea-eАВСЕавсе])")
LABEL = re.compile(r"^\s*[A-EА-Е]\s*[).:]\s+")
NUMERIC = re.compile(r"^\s*([−–-]?\d+(?:[.,]\d+)?)\s*(.*?)\s*$")

NUM = re.compile(r"\d+(?:[.,]\d+)?")      # ordinary digits; ² and ₂ are not counted


def numsig(s):
    """Sorted numbers in a question, if it has at least two: the same data reworded keeps them."""
    n = sorted(x.replace(",", ".") for x in NUM.findall(s))
    return tuple(n) if len(n) >= 2 else None


PROMPT_COT = "Шешуін қысқаша көрсетіп, дұрыс жауапты таңдаңыз."
PROMPT_LETTER = "Дұрыс жауаптың әрпін ғана жазыңыз."


# ------------------------------------------------------------------ reading
def clean(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFC", str(x)).replace("\u00a0", " ").replace("\u200b", "").replace("\r\n", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()     # keep code indentation
    return "" if s.lower() == "nan" else s


def read_sheet(path, sheet):
    try:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str, engine="openpyxl")
    except PermissionError:
        sys.exit(f"Can't open {path}. Close it in Excel and run again.")
    except ValueError as e:
        sys.exit(f"{e}. Sheets are named Questions, Examples, ...")
    df.columns = [clean(c) for c in df.columns]
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"Sheet '{sheet}' has no column(s) {missing}. Don't rename the header row.")
    df = df[COLS].copy()
    df.insert(0, "row", df.index + 2)                  # Excel row number
    for c in COLS:
        df[c] = df[c].map(clean)
    df["date"] = df["date"].str.slice(0, 10)          # '2026-11-06 00:00:00' -> '2026-11-06'
    df["grade"] = df["grade"].str.replace(r"\.0$", "", regex=True)
    return df[(df[COLS] != "").any(axis=1)].reset_index(drop=True)


def read_targets(path):
    """Targets from the Progress sheet (subject names in column A, targets in column B)."""
    try:
        from openpyxl import load_workbook
        ws = load_workbook(path, read_only=True, data_only=True)["Progress"]
        t = {}
        for a, b in ws.iter_rows(min_row=1, max_row=40, max_col=2, values_only=True):
            if isinstance(a, str) and a.strip() in SUBJECTS and isinstance(b, (int, float)):
                t[a.strip()] = int(b)
        return t if len(t) == len(SUBJECTS) else DEFAULT_TARGETS
    except Exception:
        return DEFAULT_TARGETS


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s.casefold()).replace("−", "-").replace(",", ".")


# ------------------------------------------------------------------ per-row checks
def check_rows(df, issues):
    blocked = Counter()                                # row -> number of ERROR / NOT READY issues

    def add(r, level, msg):
        issues.append(dict(row=int(r["row"]), id=r["id"] or "?", level=level, msg=msg))
        if level in ("ERROR", "NOT READY"):
            blocked[int(r["row"])] += 1

    id_count = Counter(df["id"])
    answers, ready = [], []
    for _, r in df.iterrows():
        # id and subject
        if not r["id"]:
            add(r, "ERROR", "no id")
        elif id_count[r["id"]] > 1:
            add(r, "ERROR", f"id {r['id']} is used {id_count[r['id']]} times")
        elif not re.fullmatch(r"(MAT|PHY|CHE|BIO|INF)-\d{4}", r["id"]):
            add(r, "WARNING", f"id '{r['id']}' doesn't look like MAT-0001")
        if r["subject"] not in SUBJECTS:
            add(r, "ERROR", f"subject '{r['subject']}' is not one of {', '.join(SUBJECTS)}")
        elif re.match(r"[A-Z]{3}-", r["id"]) and not r["id"].startswith(PREFIX[r["subject"]]):
            add(r, "WARNING", f"id prefix doesn't match subject {r['subject']} ({PREFIX[r['subject']]}-...)")
        # question and options
        if not r["question_kk"]:
            add(r, "ERROR", "no question")
        elif len(r["question_kk"]) < 15:
            add(r, "WARNING", "very short question")
        missing = [o[-1] for o in REQUIRED_OPTS if not r[o]]
        if missing:
            add(r, "ERROR", f"option(s) {', '.join(missing)} empty")
        if not r["option_E"] and not missing:
            add(r, "WARNING", "only 4 options; KazMMLU uses 5")
        opts = [r[o] for o in OPTS if r[o]]
        for o in OPTS:
            if r[o] and LABEL.match(r[o]):
                add(r, "ERROR", f"{o} starts with a letter label ('{r[o][:6]}...'); type only the option text")
        dup = [k for k, v in Counter(norm_text(x) for x in opts).items() if v > 1]
        if dup:
            add(r, "ERROR", "two options are the same")
        # answer
        a_raw = r["answer"]
        a = LOOKALIKE.get(a_raw, a_raw)
        if a_raw and a != a_raw and a_raw in "АВСЕавсе":
            add(r, "WARNING", f"answer typed with a Cyrillic letter '{a_raw}'; read as Latin {a}")
        if a not in LETTERS or len(a) != 1:
            add(r, "ERROR", f"answer '{a_raw}' must be one of A-E")
            a = ""
        elif not r[f"option_{a}"]:
            add(r, "ERROR", f"answer {a} points to an empty option")
        # solution
        sol = r["solution_kk"]
        if sol:                                        # a solution is optional
            m = MARK_END.search(sol)
            if not m:
                hit = MARK_ANY.search(sol)
                add(r, "ERROR", "solution must end with 'Жауабы: X'" +
                    (" (it appears, but not at the end)" if hit else ""))
            else:
                s_letter = LOOKALIKE.get(m.group(1), m.group(1))
                if m.group(1) in "АВСЕавсе":
                    add(r, "WARNING", f"'Жауабы: {m.group(1)}' uses a Cyrillic letter; exported as Latin {s_letter}")
                if a and s_letter != a:
                    add(r, "ERROR", f"solution says Жауабы: {s_letter} but the answer column says {a}")
        # source / verification
        if r["source"] not in ("original", "ai_drafted"):
            add(r, "ERROR", f"source '{r['source']}' must be original or ai_drafted")
        elif r["source"] == "ai_drafted":
            if r["verified"] != "yes":
                add(r, "NOT READY", "AI draft not verified yet (verified must be yes)")
            if not r["drafted_by"]:
                add(r, "WARNING", "ai_drafted but drafted_by is empty")
        # language
        text = " ".join([r["question_kk"], sol, *opts])
        if r["question_kk"] and not (KAZ_LETTERS & set(r["question_kk"] + sol)):
            add(r, "WARNING", "no Kazakh letters (ә ғ қ ң ө ұ ү һ і) in question or solution: Russian or untranslated?")
        mixed = sorted({t for t in TOKEN.findall(text) if HAS_LAT.search(t) and HAS_CYR.search(t)})
        if mixed:
            add(r, "WARNING", "Latin and Cyrillic letters mixed inside: " + ", ".join(mixed[:4]) +
                " (Latin i/a/o/e/c/p/x typed instead of Kazakh?)")
        # option shape
        nums = [NUMERIC.match(x) for x in opts]
        if len(opts) >= 4 and all(nums) and len({m.group(2) for m in nums}) == 1:
            vals = [float(m.group(1).replace("−", "-").replace("–", "-").replace(",", ".")) for m in nums]
            if vals != sorted(vals):
                add(r, "WARNING", "numeric options are not in increasing order")
        elif a and r[f"option_{a}"] and len(opts) >= 4:
            la = len(r[f"option_{a}"])
            other = max(len(x) for o, x in zip(OPTS, [r[o] for o in OPTS]) if x and o != f"option_{a}")
            if la > 10 and la > 1.5 * other:
                add(r, "WARNING", "the correct option is much longer than the others (a giveaway)")
        if r["grade"] and r["grade"] not in {"7", "8", "9", "10", "11"}:
            add(r, "WARNING", f"grade '{r['grade']}' is not 7-11")
        if r["difficulty"] and r["difficulty"] not in ("easy", "medium", "hard"):
            add(r, "WARNING", f"difficulty '{r['difficulty']}' is not easy/medium/hard")
        if not r["date"]:
            add(r, "WARNING", "no date")
        answers.append(a)
        ready.append(blocked[int(r["row"])] == 0)
    df["answer_norm"] = answers
    df["ready"] = ready
    return df


# ------------------------------------------------------------------ set-level checks
def tfidf(texts_a, texts_b=None):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        sys.exit("Needs scikit-learn: pip install scikit-learn")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
    vec.fit(list(texts_a) + list(texts_b or []))
    A = vec.transform(texts_a)
    return A, (vec.transform(texts_b) if texts_b is not None else None)


def duplicates(df, issues, near=0.90):
    """Exact duplicates are errors; near-duplicates are warnings and are kept in one split."""
    key = df["question_kk"].map(norm_text) + "|" + df[OPTS].apply(
        lambda r: "|".join(sorted(norm_text(x) for x in r if x)), axis=1)
    first = {}
    for i, k in enumerate(key):
        if not df.at[i, "question_kk"]:
            continue
        if k in first:
            j = first[k]
            issues.append(dict(row=int(df.at[i, "row"]), id=df.at[i, "id"] or "?", level="ERROR",
                               msg=f"same question as row {df.at[j, 'row']} ({df.at[j, 'id']})"))
            df.at[i, "ready"] = False
        else:
            first[k] = i
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    idx = [i for i in range(len(df)) if df.at[i, "question_kk"]]
    if len(idx) > 1:
        A, _ = tfidf([df.at[i, "question_kk"] for i in idx])
        S = (A @ A.T).tocoo()
        sig = [numsig(df.at[i, "question_kk"]) for i in idx]
        partners = defaultdict(list)
        for i, j, v in zip(S.row, S.col, S.data):
            if i >= j:
                continue
            same_data = sig[i] is not None and sig[i] == sig[j] and v >= 0.5
            if v >= near or same_data:
                a, b = idx[i], idx[j]
                parent[find(a)] = find(b)
                if key[a] != key[b]:
                    a, b = sorted((a, b))
                    partners[b].append((a, v, v < near))
        for b, ps in sorted(partners.items()):
            a, v, by_numbers = max(ps, key=lambda t: t[1])
            why = f"same numbers, similarity {v:.2f}" if by_numbers else f"similarity {v:.2f}"
            more = f" and {len(ps) - 1} other row(s)" if len(ps) > 1 else ""
            issues.append(dict(row=int(df.at[b, "row"]), id=df.at[b, "id"] or "?", level="WARNING",
                               msg=f"near-duplicate of row {df.at[a, 'row']} ({df.at[a, 'id']}){more}, {why}: "
                                   f"a different question, or the same template again?"))
    df["group"] = [find(i) for i in range(len(df))]
    return df


def kazmmlu_overlap(df, paths, col, issues, hard=0.90, soft=0.70):
    texts = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".csv":
            d = pd.read_csv(p, dtype=str)
        elif ext in (".jsonl", ".json"):
            d = pd.read_json(p, lines=(ext == ".jsonl"), dtype=str)
        elif ext == ".parquet":
            d = pd.read_parquet(p)
        elif ext in (".xlsx", ".xls"):
            d = pd.read_excel(p, dtype=str)
        else:
            sys.exit(f"--kazmmlu: don't know how to read {p}")
        c = col if col in d.columns else next(
            (k for k in d.columns if str(k).strip().lower() in ("question", "questions", "вопрос", "сұрақ")), None) \
            or next((k for k in d.columns if "question" in str(k).lower()), None)
        if c is None:
            sys.exit(f"--kazmmlu: no question column in {p}; columns are {list(d.columns)}. Use --kazmmlu-col NAME.")
        texts += [clean(x) for x in d[c].dropna()]
    texts = [t for t in texts if t]
    idx = [i for i in range(len(df)) if df.at[i, "question_kk"]]
    if not texts or not idx:
        return 0
    A, R = tfidf([df.at[i, "question_kk"] for i in idx], texts)
    S = (A @ R.T).toarray()
    by_sig = defaultdict(list)
    for j, t in enumerate(texts):
        if numsig(t):
            by_sig[numsig(t)].append(j)
    for k, i in enumerate(idx):
        row, q = S[k], df.at[i, "question_kk"]
        j = int(row.argmax())
        found = None
        if row[j] >= hard:
            if numsig(q) and numsig(texts[j]) and numsig(q) != numsig(texts[j]):
                found = ("WARNING", j, f"same wording as a KazMMLU question with other numbers (similarity "
                                       f"{row[j]:.2f}); better to rephrase")
            else:
                found = ("ERROR", j, f"near-copy of a KazMMLU question (similarity {row[j]:.2f})")
        elif numsig(q) in by_sig:
            jj = max(by_sig[numsig(q)], key=lambda x: row[x])
            if row[jj] >= 0.5:
                found = ("ERROR", jj, f"same numbers and similar wording as a KazMMLU question (similarity {row[jj]:.2f})")
            elif row[jj] >= 0.3:
                found = ("WARNING", jj, f"same numbers as a KazMMLU question (similarity {row[jj]:.2f}); same problem?")
        if found is None and row[j] >= soft:
            found = ("WARNING", j, f"similar wording to a KazMMLU question (similarity {row[j]:.2f}); not a rewrite?")
        if found:
            level, jj, msg = found
            issues.append(dict(row=int(df.at[i, "row"]), id=df.at[i, "id"] or "?", level=level,
                               msg=f"{msg}: «{texts[jj][:90]}»"))
            if level == "ERROR":
                df.at[i, "ready"] = False
    return len(texts)


def balance(df, issues, n_min=50):
    a = df.loc[df["ready"], "answer_norm"]
    n = len(a)
    share = {L: (a == L).mean() if n else 0.0 for L in LETTERS}
    if n >= n_min:
        off = [f"{L} {share[L]:.0%}" for L in LETTERS if not 0.15 <= share[L] <= 0.25]
        if off:
            issues.append(dict(row=0, id="(all)", level="WARNING",
                               msg="answer letters out of balance (aim for 15-25 % each): " + ", ".join(off)))
    return share


# ------------------------------------------------------------------ splits and export
def assign_splits(df, path, test_frac, issues):
    frozen = {}
    if os.path.exists(path):
        s = pd.read_csv(path, dtype=str)
        frozen = dict(zip(s["id"], s["split"]))
    h = lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32
    split = {}
    for g, grp in df[df["ready"]].groupby("group"):
        ids = list(grp["id"])
        known = {frozen[i] for i in ids if i in frozen}
        if len(known) > 1:
            issues.append(dict(row=int(grp["row"].iloc[0]), id=ids[0], level="WARNING",
                               msg=f"near-duplicates {', '.join(ids)} are in different splits; consider deleting one"))
        s = known.pop() if len(known) == 1 else ("test" if h(min(ids)) < test_frac else "train")
        for i in ids:
            split[i] = frozen.get(i, s)
    frozen.update(split)
    pd.DataFrame(sorted(frozen.items()), columns=["id", "split"]).to_csv(path, index=False)
    df["split"] = df["id"].map(split).fillna("")
    return df


def final_solution(r):
    """Solution with the closing marker normalised to 'Жауабы: X' (Latin letter)."""
    return MARK_END.sub(f"Жауабы: {r['answer_norm']}", r["solution_kk"]).rstrip()


def prompt_text(r, instruction):
    lines = [r["question_kk"], ""] + [f"{L}) {r[f'option_{L}']}" for L in LETTERS if r[f"option_{L}"]]
    return "\n".join(lines + ["", instruction])


def export(df, out):
    ok = df[df["ready"]]
    fields = [c for c in COLS if c not in ("answer",)] + ["split"]
    with open(f"{out}/questions_clean.jsonl", "w", encoding="utf-8") as f:
        for _, r in ok.iterrows():
            d = {c: r[c] for c in fields}
            d["answer"], d["solution_kk"] = r["answer_norm"], final_solution(r)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    train = ok[ok["split"] == "train"]
    with open(f"{out}/sft_train_cot.jsonl", "w", encoding="utf-8") as f1, \
            open(f"{out}/sft_train_letter.jsonl", "w", encoding="utf-8") as f2:
        for _, r in train.iterrows():
            if r["solution_kk"]:
                f1.write(json.dumps({"id": r["id"], "messages": [
                    {"role": "user", "content": prompt_text(r, PROMPT_COT)},
                    {"role": "assistant", "content": final_solution(r)}]}, ensure_ascii=False) + "\n")
            f2.write(json.dumps({"id": r["id"], "messages": [
                {"role": "user", "content": prompt_text(r, PROMPT_LETTER)},
                {"role": "assistant", "content": r["answer_norm"]}]}, ensure_ascii=False) + "\n")
    test = ok[ok["split"] == "test"]
    pd.DataFrame({"id": test["id"], "subject": test["subject"], "topic": test["topic"], "question": test["question_kk"],
                  **{L: test[f"option_{L}"] for L in LETTERS}, "answer": test["answer_norm"]}) \
        .to_csv(f"{out}/test_own.csv", index=False, encoding="utf-8-sig")
    return len(ok), len(train), len(test)


# ------------------------------------------------------------------ report
def write_report(df, issues, share, targets, counts, out, n_ref, sheet):
    lv = Counter(i["level"] for i in issues)
    rows_err = len({i["row"] for i in issues if i["level"] == "ERROR" and i["row"]})
    n_ok, n_train, n_test = counts
    L = [f"# Question check: sheet '{sheet}'", "",
         f"- Rows with content: **{len(df)}**",
         f"- Ready for training: **{n_ok}** (train {n_train}, held-out test {n_test})",
         f"- Rows with errors: **{rows_err}** · AI drafts waiting for your check: **{lv['NOT READY']}** · "
         f"warnings: **{lv['WARNING']}**",
         f"- KazMMLU overlap check: " + (f"against {n_ref} KazMMLU questions" if n_ref else "not run (use --kazmmlu)"),
         "", "## By subject", "", "| Subject | Target | Rows | Ready | Original | AI-drafted |", "|---|---|---|---|---|---|"]
    for s in SUBJECTS:
        d = df[df["subject"] == s]
        L.append(f"| {s} | {targets.get(s, '')} | {len(d)} | {int(d['ready'].sum())} | "
                 f"{(d['source'] == 'original').sum()} | {(d['source'] == 'ai_drafted').sum()} |")
    L.append(f"| **total** | {sum(targets.values())} | {len(df)} | {n_ok} | {(df['source'] == 'original').sum()} | "
             f"{(df['source'] == 'ai_drafted').sum()} |")
    nosol = int((df.loc[df["ready"], "solution_kk"] == "").sum())
    if nosol:
        L += ["", f"Ready questions without a worked solution: {nosol} (trained on the answer letter only)."]
    ai = df[df["source"] == "ai_drafted"]
    if len(ai):
        L += ["", f"AI drafts you corrected: {(ai['corrections'] != '').sum()} of {len(ai)} "
                  f"({(ai['corrections'] != '').mean():.0%})."]
    L += ["", "## Answer letters (ready questions)", "",
          " · ".join(f"{k} {v:.0%}" for k, v in share.items()) + "  (aim for 15-25 % each)", ""]
    for level in ("ERROR", "NOT READY", "WARNING"):
        items = sorted([i for i in issues if i["level"] == level], key=lambda i: (i["row"], i["msg"]))
        if not items:
            continue
        L += [f"## {level} ({len(items)})", "", "| Excel row | id | Problem |", "|---|---|---|"]
        L += [f"| {i['row'] or ''} | {i['id']} | {i['msg'].replace('|', '/')} |" for i in items]
        L.append("")
    open(f"{out}/report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Check and export the Kazakh STEM question workbook.")
    ap.add_argument("workbook")
    ap.add_argument("--sheet", default="Questions")
    ap.add_argument("--out", default=None, help="output folder (default export/, or export_<sheet>/)")
    ap.add_argument("--test-frac", type=float, default=0.2, help="share held out as your own test set")
    ap.add_argument("--kazmmlu", nargs="*", default=[], help="KazMMLU files to check overlap against")
    ap.add_argument("--kazmmlu-col", default="question")
    args = ap.parse_args()
    out = args.out or ("export" if args.sheet == "Questions" else f"export_{args.sheet.lower().replace(' ', '_')}")
    os.makedirs(out, exist_ok=True)

    df = read_sheet(args.workbook, args.sheet)
    if df.empty:
        sys.exit(f"Sheet '{args.sheet}' has no questions yet.")
    issues = []
    df = check_rows(df, issues)
    df = duplicates(df, issues)
    n_ref = kazmmlu_overlap(df, args.kazmmlu, args.kazmmlu_col, issues) if args.kazmmlu else 0
    share = balance(df, issues)
    df = assign_splits(df, f"{out}/splits.csv", args.test_frac, issues)
    counts = export(df, out)
    targets = read_targets(args.workbook)
    write_report(df, issues, share, targets, counts, out, n_ref, args.sheet)

    lv = Counter(i["level"] for i in issues)
    print(f"{len(df)} rows · ready {counts[0]} (train {counts[1]}, test {counts[2]}) · "
          f"errors {lv['ERROR']} · not ready {lv['NOT READY']} · warnings {lv['WARNING']}")
    for i in sorted(issues, key=lambda i: (i["level"] != "ERROR", i["row"]))[:25]:
        print(f"  {i['level']:9s} row {i['row'] or '-':>4} {i['id']:9s} {i['msg']}")
    if len(issues) > 25:
        print(f"  ... {len(issues) - 25} more")
    print(f"Full report: {out}/report.md")
    sys.exit(1 if lv["ERROR"] else 0)


if __name__ == "__main__":
    main()
