"""
Figures for the full Sarab run (read-only; saved results, no API calls).

    python3 analysis/figures.py      # writes analysis/results/figs/*.png and *.pdf

fig_accuracy      accuracy per model and mode with 95% image-clustered bootstrap CIs
fig_trust_caption CATR (share of answers that follow the misleading caption) for Sway and False
fig_difficulty    share of images solved by k of the 6 models, per mode
fig_blank         Blank detection rates (MCDR, OEDR, UDR) and control error, Wilson 95% CIs
Colours are the Okabe-Ito colour-blind-safe palette; no Arabic text appears in the figures.
"""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis as ma

FIG = ma.OUT / "figs"; FIG.mkdir(exist_ok=True)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]       # Okabe-Ito
MODELS = [ma.SHORT[m] for m in ma.MODEL_ORDER]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 120})


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=200, bbox_inches="tight"); fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", FIG / (name + ".png"))


def headline():
    h = {}
    for r in csv.DictReader(open(ma.OUT / "main_headline.csv", encoding="utf-8")):
        h[(r["mode"], r["model"])] = {k: float(v) for k, v in r.items() if k not in ("mode", "model")}
    return h


def bars(ax, groups, series, vals, los, his, ylabel, width=0.13):
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = np.array([vals[g][i] for g in groups]) * 100
        err = np.array([[v[j] - los[groups[j]][i] * 100 for j in range(len(groups))], [his[groups[j]][i] * 100 - v[j] for j in range(len(groups))]])
        ax.bar(x + (i - (len(series) - 1) / 2) * width, v, width, yerr=err, capsize=1.5, color=COLORS[i], label=s, error_kw={"lw": 0.7})
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel(ylabel)


def fig_accuracy(h):
    modes = list(ma.MODES.values()); fig, ax = plt.subplots(figsize=(7.2, 3.3))
    vals = {m: [h[(m, s)]["acc_lenient"] for s in MODELS] for m in modes}
    lo = {m: [h[(m, s)]["acc_lo"] for s in MODELS] for m in modes}; hi = {m: [h[(m, s)]["acc_hi"] for s in MODELS] for m in modes}
    bars(ax, modes, MODELS, vals, lo, hi, "Accuracy (%)")
    ax.axhline(50, color="k", lw=0.8, ls="--"); ax.text(3.45, 51, "chance", ha="right", fontsize=7); ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    ax.set_title("Accuracy by mode (95% CI, resampling images)", y=1.28, fontsize=9); save(fig, "fig_accuracy")


def fig_trust(h):
    modes = ["Sway", "False"]; fig, ax = plt.subplots(figsize=(4.6, 3.1))
    vals = {m: [1 - h[(m, s)]["acc_lenient"] for s in MODELS] for m in modes}
    lo = {m: [1 - h[(m, s)]["acc_hi"] for s in MODELS] for m in modes}; hi = {m: [1 - h[(m, s)]["acc_lo"] for s in MODELS] for m in modes}
    bars(ax, modes, MODELS, vals, lo, hi, "Answers that follow the wrong caption (%)", width=0.125)
    ax.axhline(50, color="k", lw=0.8, ls="--"); ax.set_ylim(0, 100)
    ax.legend(ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.33), frameon=False)
    ax.set_title("Trust in a misleading caption (CATR, lower is better)", y=1.40, fontsize=9); save(fig, "fig_trust_caption")


def fig_difficulty(data):
    modes = list(ma.MODES.items()); fig, ax = plt.subplots(figsize=(5.2, 2.8)); left = np.zeros(len(modes))
    shares = []
    for mode, _ in modes:
        solved = defaultdict(int)
        for m in ma.MODEL_ORDER:
            for img, a in ma.per_image(data[(m, mode)], "lenient").items():
                solved[img] += int(a[0] + a[1] == 2)
        c = Counter(solved.values()); n = len(solved); shares.append([c.get(k, 0) / n for k in range(7)])
    shares = np.array(shares) * 100
    cmap = plt.get_cmap("viridis_r", 7)
    for k in range(7):
        ax.barh(range(len(modes)), shares[:, k], left=left, color=cmap(k), label=("%d models" % k) if k != 1 else "1 model")
        left += shares[:, k]
    ax.set_yticks(range(len(modes))); ax.set_yticklabels([n for _, n in modes]); ax.invert_yaxis(); ax.set_xlabel("Share of images (%)")
    ax.legend(ncol=7, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, handlelength=1, columnspacing=0.8)
    ax.set_title("Images solved (both questions right) by k of 6 models", y=1.3, fontsize=9); save(fig, "fig_difficulty")


def fig_blank(data):
    rates = {}
    for m in ma.MODEL_ORDER:
        rows = data[(m, 5)]; rates[m] = {}
        for label, sel, flip in (("MCDR", lambda r: r["condition"] == "mcdr" and r["is_control"] == "0", False),
                                 ("OEDR", lambda r: r["condition"] == "oedr", False),
                                 ("UDR", lambda r: r["condition"] == "udr", False),
                                 ("Control error", lambda r: r["condition"] == "mcdr" and r["is_control"] == "1", True)):
            rs = [r for r in rows if sel(r)]; k = sum((1 - int(r["is_correct"])) if flip else int(r["is_correct"]) for r in rs); n = len(rs)
            lo, hi = ma.wilson(k, n); rates[m][label] = (k / n, lo, hi)
    labels = ["MCDR", "OEDR", "UDR", "Control error"]; fig, ax = plt.subplots(figsize=(7.2, 3.1)); x = np.arange(len(labels)); w = 0.13
    for i, m in enumerate(ma.MODEL_ORDER):
        v = [rates[m][k][0] * 100 for k in labels]; lo = [rates[m][k][1] * 100 for k in labels]; hi = [rates[m][k][2] * 100 for k in labels]
        ax.bar(x + (i - 2.5) * w, v, w, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=1.5, color=COLORS[i], label=ma.SHORT[m], error_kw={"lw": 0.7})
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Rate (%), 30 items each"); ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    ax.set_title("Blank: absent-answer detection (Wilson 95% CI)", y=1.28, fontsize=9); save(fig, "fig_blank")


if __name__ == "__main__":
    h = headline(); data = ma.load()
    fig_accuracy(h); fig_trust(h); fig_difficulty(data); fig_blank(data)
