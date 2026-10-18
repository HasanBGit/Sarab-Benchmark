"""
Error types, accuracy by category, and position bias in Blank. Read-only: saved results, no API calls.

    python3 analysis/error_types_position.py

Writes analysis/results/error_types.{csv,md}, category_heatmap.csv, blank_position.{csv,md} and
analysis/results/figs/fig_error_types, fig_category_heatmap, fig_blank_position (.png and .pdf).

1. Error types (Ground, Sway, False, Clash): every wrong task is one of
     false accept  - the right answer was no, the model said yes (followed a wrong claim or caption)
     false reject  - the right answer was yes, the model said no (denied a true claim)
     declined      - the model refused to answer (for example, would not name a person's origin)
     no answer     - an API error, an empty reply, or a reply that starts with neither yes nor no
2. Accuracy by category: model x category cells for Ground, Sway and False (tasks right / tasks).
3. Control errors in Blank: of the controls answered wrongly, how many picked the none option and how many another item.
   Position bias (Blank, MCDR + control, 60 four-option tasks per model): the chosen letter against the
   letter of the correct answer, plus how often the model picks the 'none' option by position.
   Chi-square goodness-of-fit of the chosen letters against the answer-key letters (descriptive: 60 tasks).
"""
import csv, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis as ma

OUT, FIG = ma.OUT, ma.OUT / "figs"
NONE_AR = "لا توجد إجابة صحيحة"
SHORT = [ma.SHORT[m] for m in ma.MODEL_ORDER]
COLORS = {"false accept": "#D55E00", "false reject": "#0072B2", "declined": "#E69F00"}   # no-answer is under 1% in every cell (1 reply)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 120})


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", FIG / (name + ".png"))


def error_types(data):
    rows, table = [], {}
    for mode in ma.MODES.values():
        k = [m for m, v in ma.MODES.items() if v == mode][0]
        for m in ma.MODEL_ORDER:
            c = Counter(); n = 0
            for r in data[(m, k)]:
                n += 1
                _, p = ma.lenient_parse(r)
                if p in ("yes", "no"):
                    if ma.is_ok(r["ground_truth"], p):
                        continue
                    c["false accept" if r["ground_truth"] == ma.NO else "false reject"] += 1
                elif p == "declined":
                    c["declined"] += 1
                else:
                    c["no answer"] += 1
            table[(mode, ma.SHORT[m])] = (c, n)
            rows.append([mode, ma.SHORT[m], n, c["false accept"], c["false reject"], c["declined"], c["no answer"]])
    with open(OUT / "error_types.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["mode", "model", "tasks", "false_accept", "false_reject", "declined", "no_answer"]); w.writerows(rows)
    md = ["# Error types per model and mode (share of all tasks in the mode)\n",
          "| Mode | Model | Tasks | Wrong | False accept | False reject | Declined | No answer |", "|" + "---|" * 8]
    for r in rows:
        wrong = sum(r[3:])
        md.append("| %s | %s | %d | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |" % (
            (r[0], r[1], r[2], 100 * wrong / r[2]) + tuple(100 * x / r[2] for x in r[3:])))
    (OUT / "error_types.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return table


def fig_error_types(table):
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True); axes[0].invert_yaxis()
    for ax, mode in zip(axes, ma.MODES.values()):
        bottom = np.zeros(len(SHORT))
        for kind, col in COLORS.items():
            v = np.array([100 * table[(mode, s)][0][kind] / table[(mode, s)][1] for s in SHORT])
            ax.barh(np.arange(len(SHORT)), v, left=bottom, color=col, label=kind, height=0.7)
            bottom += v
        ax.set_title(mode, fontsize=9); ax.set_yticks(np.arange(len(SHORT)))
        ax.set_yticklabels(SHORT, fontsize=7.5)
        ax.tick_params(axis="y", labelleft=(mode == "Ground"))
        ax.set_xlim(0, 80); ax.set_xlabel("% of tasks wrong"); ax.grid(axis="y", alpha=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.55, 1.06), frameon=False)
    fig.tight_layout(); save(fig, "fig_error_types")


def category_heatmap(data):
    cells, rows = {}, []
    for k, mode in list(ma.MODES.items())[:3]:
        agg = defaultdict(lambda: [0, 0]); imgs = defaultdict(set)
        for m in ma.MODEL_ORDER:
            for r in data[(m, k)]:
                _, p = ma.lenient_parse(r)
                a = agg[(r["category"], ma.SHORT[m])]
                a[0] += ma.is_ok(r["ground_truth"], p) if p in ("yes", "no") else 0
                a[1] += 1; imgs[r["category"]].add(r["image_id"])
        cats = sorted(imgs, key=lambda c: -len(imgs[c]))
        cells[mode] = (cats, {c: len(imgs[c]) for c in cats}, agg)
        for c in cats:
            for s in SHORT:
                rows.append([mode, c, len(imgs[c]), s, agg[(c, s)][0], agg[(c, s)][1]])
    with open(OUT / "category_heatmap.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["mode", "category", "images", "model", "tasks_right", "tasks"]); w.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), gridspec_kw={"width_ratios": [4, 5, 5], "wspace": 0.75})
    for ax, mode in zip(axes, ma.MODES.values()):
        cats, nimg, agg = cells[mode]
        mat = np.array([[100 * agg[(c, s)][0] / agg[(c, s)][1] for s in SHORT] for c in cats])
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, "%.0f" % mat[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if mat[i, j] < 60 else "black")
        ax.set_xticks(range(len(SHORT))); ax.set_xticklabels([s.split()[0] for s in SHORT], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(["%s (%d)" % (c, nimg[c]) for c in cats], fontsize=7.5)
        ax.set_title(mode, fontsize=9); ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_label("Accuracy (%)", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, "fig_category_heatmap")


def blank_position(data):
    out, md = [], ["# Blank: position of the chosen option (MCDR + control, 60 four-option tasks per model)\n",
                   "Chosen letters are compared with the letters of the answer key. Control errors are the 30 control tasks (true answer listed) answered wrongly; 'None picked' is the part that chose the 'no correct answer' option (false abstention in the strict sense), 'Other item' the part that chose a wrong named option.\n",
                   "| Model | A | B | C | D | Key A/B/C/D | chi2 p | Control errors | None picked | Other item |", "|" + "---|" * 10]
    key = None
    stats = {}
    for m in ma.MODEL_ORDER:
        rows = [r for r in data[(m, 5)] if r["condition"] == "mcdr"]
        if key is None:
            key = Counter(r["correct_letter"] for r in rows)
        ch = Counter(r["parsed_letter"] for r in rows if r["parsed_letter"] in "ABCD" and r["parsed_letter"])
        n = sum(ch.values()); kc = np.array([key[x] for x in "ABCD"], float)
        exp = kc / kc.sum() * n
        p = chisquare([ch[x] for x in "ABCD"], exp).pvalue if n else float("nan")
        none_picked = wrong_item = 0
        for r in [x for x in rows if x["is_control"] == "1"]:
            L = r["parsed_letter"]
            if L not in ("A", "B", "C", "D"):
                continue
            opts = {o.split("=", 1)[0].strip(): o.split("=", 1)[1] for o in r["options_shown"].split(" | ") if "=" in o}
            is_none = NONE_AR in opts.get(L, "")
            if r["is_correct"] == "1":
                continue
            if is_none:
                none_picked += 1
            else:
                wrong_item += 1
        first = ch["A"]
        stats[ma.SHORT[m]] = (ch, n, p)
        md.append("| %s | %d | %d | %d | %d | %s | %.2f | %d | %d | %d |" % (
            ma.SHORT[m], ch["A"], ch["B"], ch["C"], ch["D"], "/".join(str(key[x]) for x in "ABCD"), p, none_picked + wrong_item, none_picked, wrong_item))
        out.append([ma.SHORT[m], ch["A"], ch["B"], ch["C"], ch["D"], round(p, 3), none_picked + wrong_item, none_picked, wrong_item])
    (OUT / "blank_position.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT / "blank_position.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["model", "A", "B", "C", "D", "chi2_p", "control_errors", "none_picked", "other_item"]); w.writerows(out)
    kc = np.array([key[x] for x in "ABCD"], float)
    fig, ax = plt.subplots(figsize=(3.4, 2.7)); x = np.arange(4)
    for i, s in enumerate(SHORT):
        ch, n, _ = stats[s]
        ax.plot(x, [100 * ch[l] / n for l in "ABCD"], marker="o", ms=3, lw=1,
                color=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"][i], label=s)
    ax.plot(x, 100 * kc / kc.sum(), color="k", lw=1.6, ls="--", label="Answer key")
    ax.set_xticks(x); ax.set_xticklabels(list("ABCD")); ax.set_ylabel("Chosen (%)"); ax.set_xlabel("Option letter")
    ax.set_ylim(0, 60); ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    save(fig, "fig_blank_position")


def main():
    data = ma.load()
    table = error_types(data)
    fig_error_types(table)
    category_heatmap(data)
    blank_position(data)
    for f in ("error_types.md", "blank_position.md"):
        print((OUT / f).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
