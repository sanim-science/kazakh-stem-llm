"""Two-panel forest plot: change in accuracy after fine-tuning (3-seed mean, 95 % bootstrap interval over questions),
held-out questions vs KazMMLU Kazakh STEM, for Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OWN, KAZ = "#2a78d6", "#eb6834"
ROWS = [("All five subjects", "own_test", "kazmmlu_stem"),
        ("Physics", "own_test / physics", "kazmmlu_stem / Physics (High School in kaz)"),
        ("Informatics", "own_test / informatics", "kazmmlu_stem / Informatics (High School in kaz)"),
        ("Chemistry", "own_test / chemistry", "kazmmlu_stem / Chemistry (High School in kaz)"),
        ("Math", "own_test / math", "kazmmlu_stem / Math (High School in kaz)"),
        ("Biology", "own_test / biology", "kazmmlu_stem / Biology (High School in kaz)")]
PANELS = [("Llama-3.1-8B-Instruct", "three_seed_summary_llama.csv"),
          ("Qwen2.5-7B-Instruct", "three_seed_summary_qwen.csv")]
XMIN, XMAX = -15, 58

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for p, (ax, (title, path)) in enumerate(zip(axes, PANELS)):
    get = pd.read_csv(path).set_index("test")
    ax.set_facecolor(SURF)
    for i, (label, own_key, kaz_key) in enumerate(ROWS):
        y0 = -i * 1.0
        for key, color, dy in ((own_key, OWN, +0.17), (kaz_key, KAZ, -0.17)):
            r = get.loc[key]
            y = y0 + dy
            ax.plot([r.ci_low * 100, r.ci_high * 100], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
            for s in str(r.seed_changes).split(" / "):
                ax.plot(float(s), y, "o", ms=4.5, mfc=SURF, mec=color, mew=1.2, zorder=3)
            ax.plot(r.change * 100, y, "o", ms=8.5, color=color, mec=SURF, mew=2, zorder=4)
            ax.text(r.ci_high * 100 + 1.2, y, f"{r.change * 100:+.1f}", va="center", ha="left", fontsize=9,
                    color=INK if i == 0 else INK2, fontweight="bold" if i == 0 else "normal")
        if p == 0:
            ax.text(-0.02, y0, label, transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=10,
                    color=INK, fontweight="bold" if i == 0 else "normal")
        if i == 0:
            ax.axhline(y0 - 0.55, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    for x in range(-10, XMAX, 10):
        if x:
            ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-len(ROWS) + 0.4, 0.75)
    ax.set_yticks([])
    ax.set_xticks(range(-10, XMAX, 10))
    ax.set_xticklabels([f"{x:+d}" if x else "0" for x in range(-10, XMAX, 10)], color=INK2)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, length=0)
fig.supxlabel("change in accuracy after fine-tuning (percentage points)", color=INK2, fontsize=10, y=0.025)
handles = [Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=OWN, mec=SURF, mew=1.5, label="Held-out questions (413)"),
           Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=KAZ, mec=SURF, mew=1.5, label="KazMMLU Kazakh STEM (4,174)"),
           Line2D([], [], marker="o", ls="", ms=4.5, mfc=SURF, mec=INK2, mew=1.2, label="Each training seed")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.885), ncol=3, frameon=False, fontsize=9,
           labelcolor=INK2, handlelength=2.2, columnspacing=1.6)
fig.text(0.015, 0.97, "Fine-tuning improves held-out questions, not KazMMLU",
         fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.015, 0.922, "Change after fine-tuning, mean of 3 training seeds; lines are 95% bootstrap intervals over questions",
         fontsize=9.5, color=INK2, ha="left", va="top")
fig.subplots_adjust(left=0.115, right=0.985, top=0.79, bottom=0.11, wspace=0.06)
fig.savefig("two_model_change.png", facecolor=SURF)
print("saved two_model_change.png")
