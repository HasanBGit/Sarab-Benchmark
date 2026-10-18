"""
Write the LaTeX tables and a numbers file for the paper from the saved results. Read-only; no API calls.

    python3 analysis/paper_tables.py [OUTPUT_DIR]      # default: ../Sarab_Arabic_Visual_Hallucination_Benchmark/tables

Every number in these tables comes from modes/*/results_*.csv through main_analysis.py, so the paper cannot drift from the data.
"""
import csv, itertools, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis as ma

ROOT = ma.ROOT
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Sarab_Arabic_Visual_Hallucination_Benchmark" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ma.MODEL_ORDER
SHORT = ma.SHORT
MODES = ma.MODES                                   # 1..4 -> Ground, Sway, False, Clash


def pct(x, d=1):
    return ("%." + str(d) + "f") % (100 * x)


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", OUT / name)


def headline():
    h = {}
    for r in csv.DictReader(open(ma.OUT / "main_headline.csv", encoding="utf-8")):
        h[(r["mode"], r["model"])] = {k: float(v) for k, v in r.items() if k not in ("mode", "model")}
    return h


def best_markers(vals):
    """index set of the best value (bold in the table)."""
    m = max(vals)
    return {i for i, v in enumerate(vals) if abs(v - m) < 1e-12}


def main():
    h = headline(); data = ma.load(); mode_names = list(MODES.values())

    # ---- T: headline accuracy with CIs ----
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Model & Ground & Sway & False & Clash \\", r"\midrule"]
    cols = {m: [h[(m, SHORT[k])]["acc_lenient"] for k in MODELS] for m in mode_names}
    best = {m: best_markers(cols[m]) for m in mode_names}
    for i, k in enumerate(MODELS):
        cells = []
        for m in mode_names:
            r = h[(m, SHORT[k])]; txt = "%s\\,[%s,\\,%s]" % (pct(r["acc_lenient"]), pct(r["acc_lo"], 0), pct(r["acc_hi"], 0))
            cells.append(("\\textbf{%s}" % txt) if i in best[m] else txt)
        lines.append("%s & %s \\\\" % (SHORT[k], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    pass  # replaced by the wide tables below

    # ---- T: PhD index, yes-bias, CATR, paired consistency (appendix) ----
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Mode & Model & PhD Idx & Yes-bias & CATR & Paired \\", r"\midrule"]
    for m in mode_names:
        for j, k in enumerate(MODELS):
            r = h[(m, SHORT[k])]
            catr = pct(1 - r["acc_lenient"], 0) + r"\%" if m in ("Sway", "False") else "n/a"
            lead = ("\\multirow{6}{*}{%s}" % m) if j == 0 else ""
            lines.append("%s & %s & %.3f & %s\\%% & %s & %s\\%% \\\\" % (lead, SHORT[k], r["phd"], pct(r["yes_bias"], 0), catr, pct(r["paired_consistency"], 0)))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    pass  # replaced by the wide tables below

    # ---- T: significance summary ----
    ok = {mode: {k: {r["task_id"]: ma.is_ok(r["ground_truth"], ma.lenient_parse(r)[1]) for r in data[(k, mode)]} for k in MODELS} for mode in MODES}
    wins = {mode: {k: [0, 0] for k in MODELS} for mode in MODES}                       # [beats, loses_to] (Holm p<0.05)
    for mode in MODES:
        pairs, ps, info = [], [], []
        for a, b in itertools.combinations(MODELS, 2):
            common = set(ok[mode][a]) & set(ok[mode][b])
            ab = sum(ok[mode][a][t] and not ok[mode][b][t] for t in common); ba = sum(ok[mode][b][t] and not ok[mode][a][t] for t in common)
            p = 1.0 if ab + ba == 0 else binomtest(ab, ab + ba, 0.5).pvalue
            pairs.append((a, b)); ps.append(p); info.append((ab, ba))
        adj = ma.holm(np.array(ps))
        for (a, b), (ab, ba), q in zip(pairs, info, adj):
            if q < 0.05:
                w, l = (a, b) if ab > ba else (b, a)
                wins[mode][w][0] += 1; wins[mode][l][1] += 1
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Model & Ground & Sway & False & Clash \\", r"\midrule"]
    for k in MODELS:
        lines.append("%s & %s \\\\" % (SHORT[k], " & ".join("%d / %d" % tuple(wins[m_][k]) for m_ in MODES)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    pass  # replaced by the wide tables below

    # ---- T: Blank ----
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Model & MCDR & OEDR & UDR & Control err. \\", r"\midrule"]
    blank = {}
    for k in MODELS:
        rows = data[(k, 5)]
        def cell(sel, flip=False):
            rs = [r for r in rows if sel(r)]; v = sum((1 - int(r["is_correct"])) if flip else int(r["is_correct"]) for r in rs); n = len(rs)
            lo, hi = ma.wilson(v, n); return v, n, "%s\\,[%s,\\,%s]" % (pct(v / n, 0), pct(lo, 0), pct(hi, 0))
        a = cell(lambda r: r["condition"] == "mcdr" and r["is_control"] == "0")
        b = cell(lambda r: r["condition"] == "oedr")
        c = cell(lambda r: r["condition"] == "udr")
        d = cell(lambda r: r["condition"] == "mcdr" and r["is_control"] == "1", flip=True)
        blank[k] = (a, b, c, d)
        lines.append("%s & %s & %s & %s & %s \\\\" % (SHORT[k], a[2], b[2], c[2], d[2]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    pass  # replaced by the wide tables below

    # ---- T: dataset statistics ----
    ref = {mode: data[(MODELS[0], mode)] for mode in list(MODES) + [5]}
    cats = {mode: Counter() for mode in ref}
    imgs = {mode: set() for mode in ref}
    for mode, rows in ref.items():
        for r in rows:
            imgs[mode].add(r["image_id"])
    for mode in MODES:
        seen = {}
        for r in ref[mode]:
            seen[r["image_id"]] = r.get("category", "")
        cats[mode] = Counter(seen.values())
    seen5 = {r["image_id"]: r["image_id"].split("_")[0] for r in ref[5]}
    cats[5] = Counter(seen5.values())
    order = ["cuisine", "attire", "architecture", "objects", "script"]
    lines = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrrr}", r"\toprule", r"Mode & Images & Tasks & Cuis. & Attire & Arch. & Obj. & Script \\", r"\midrule"]
    names = {1: "Ground", 2: "Sway", 3: "False", 4: "Clash", 5: "Blank"}
    for mode in (1, 2, 3, 5):
        lines.append("%s & %d & %d & %s \\\\" % (names[mode], len(imgs[mode]), len(ref[mode]), " & ".join(str(cats[mode].get(c, 0)) for c in order)))
    lines.append("Clash & %d & %d & & & & & \\\\" % (len(imgs[4]), len(ref[4])))
    lines += [r"\midrule", "Total & %d & %d & & & & & \\\\" % (sum(len(imgs[m]) for m in imgs), sum(len(ref[m]) for m in ref)), r"\bottomrule", r"\end{tabular*}"]
    write("t_dataset.tex", "\n".join(lines) + "\n")

    # ---- T: cost and tokens ----
    cost = {}
    for k in MODELS:
        usd = pt = ct = rt = n = 0
        for mode in list(MODES) + [5]:
            for r in data[(k, mode)]:
                try:
                    usd += float(r["cost_usd"]); pt += int(r["prompt_tokens"]); ct += int(r["completion_tokens"]); rt += int(r["reasoning_tokens"]); n += 1
                except ValueError:
                    pass
        cost[k] = (n, usd, pt, ct, rt)
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Model & Calls & Input tok. & Output tok. & Reasoning tok. & Cost (USD) \\", r"\midrule"]
    for k in MODELS:
        n, usd, pt, ct, rt = cost[k]
        lines.append("%s & %s & %.2fM & %.2fM & %.2fM & %.2f \\\\" % (SHORT[k], format(n, ","), pt / 1e6, ct / 1e6, rt / 1e6, usd))
    tot = sum(v[1] for v in cost.values())
    lines += [r"\midrule", r"Total & & & & & %.2f \\" % tot, r"\bottomrule", r"\end{tabular}"]
    pass  # replaced by the wide tables below

    # ---- T: Ground ranking under item removal ----
    sens = ma.OUT / "validity_sensitivity.md"
    # recompute directly (same rules as validity_sensitivity.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validity_sensitivity as vs
    refg = data[(MODELS[0], 1)]
    overlap = {r["image_id"] for r in refg if (r["gt_ar"], r["h_item_ar"]) in vs.OVERLAP}
    person = {r["image_id"] for r in refg if vs.PERSON.search(r["question_text"])}
    scen = [("All 270 images", set()), ("Without %d overlapping-label images" % len(overlap), overlap),
            ("Also without %d person-nationality images" % len(person - overlap), overlap | person)]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Model & All 270 & \shortstack{Without %d\\overlapping} & \shortstack{Also without %d\\person images} \\" % (len(overlap), len(person - overlap)), r"\midrule"]
    cols = [[vs.acc([r for r in data[(k, 1)] if r["image_id"] not in ex]) for k in MODELS] for _, ex in scen]
    for n_, k in enumerate(MODELS):
        lines.append("%s & %s \\\\" % (SHORT[k], " & ".join(pct(c[n_]) for c in cols)))
    lines.append(r"\midrule")
    lines.append("Images kept & %s \\\\" % " & ".join(str(270 - len(ex)) for _, ex in scen))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t_sensitivity.tex", "\n".join(lines) + "\n")

    # ---- T: category accuracy, pooled over the six models ----
    cat_rows = list(csv.DictReader(open(ma.OUT / "category_accuracy.csv", encoding="utf-8")))
    pooled = defaultdict(lambda: [0, 0])
    for r in cat_rows:
        pooled[(r["mode"], r["category"])][0] += int(r["tasks_right"]); pooled[(r["mode"], r["category"])][1] += int(r["tasks"])
    imgs_by = defaultdict(set)
    for mode, name in MODES.items():
        for r in data[(MODELS[0], mode)]:
            imgs_by[(name, r.get("category", ""))].add(r["image_id"])
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Category & Ground & Sway & False \\", r"\midrule"]
    for c in ["cuisine", "attire", "architecture", "objects", "script"]:
        cells = []
        for name in ("Ground", "Sway", "False"):
            v = pooled.get((name, c))
            cells.append("%s\\,(%d)" % (pct(v[0] / v[1], 0) + "\\%", len(imgs_by[(name, c)])) if v else "n/a")
        lines.append("%s & %s \\\\" % (c.capitalize(), " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t_category.tex", "\n".join(lines) + "\n")

    # ---- T: explanation-based measures ----
    import re as _re, explanation_analysis as ea
    udr = defaultdict(int); udr_n = defaultdict(int)
    for r in csv.DictReader(open(Path(__file__).resolve().parent / "udr_full_read_labels.csv", encoding="utf-8")):
        udr_n[r["model"]] += 1; udr[r["model"]] += r["label"] == "K"
    contra = defaultdict(int)
    for r in csv.DictReader(open(Path(__file__).resolve().parent / "contradiction_labels.csv", encoding="utf-8")):
        if r["label"] == "TRUE":
            contra[r["model"]] += 1
    foreign = _re.compile(r"[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff\uac00-\ud7af]")
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Model & \shortstack{UDR: names\\real answer} & \shortstack{Answer vs.\\explanation} & \shortstack{Cites\\caption} & \shortstack{Foreign\\script} \\", r"\midrule"]
    ev = {}
    for k in MODELS:
        rs23 = [r for mode in (2, 3) for r in data[(k, mode)]]
        cite = sum(bool(ea.CITE.search(r["raw_response"])) for r in rs23) / len(rs23)
        allr = [r for mode in list(MODES) + [5] for r in data[(k, mode)]]
        fs = sum(bool(foreign.search(r["raw_response"])) for r in allr)
        ev[k] = (udr[k], contra[SHORT[k]], pct(cite, 0), fs)
    lines += [r"\bottomrule", r"\end{tabular}"]
    pass  # replaced by the wide tables below


    # ---- wide tables (full text width) ----
    FILL = r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}%s}"
    # accuracy with CIs, and how many models each model beats / is beaten by
    cols = {m: [h[(m, SHORT[k])]["acc_lenient"] for k in MODELS] for m in mode_names}
    best = {m: best_markers(cols[m]) for m in mode_names}
    lines = [FILL % ("l" + "c" * 8), r"\toprule",
             r"& \multicolumn{4}{c}{Accuracy, \% (95\% CI)} & \multicolumn{4}{c}{Beats / beaten by} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}", "Model" + " & Ground & Sway & False & Clash" * 2 + r" \\", r"\midrule"]
    for i, k in enumerate(MODELS):
        acc = []
        for m in mode_names:
            r = h[(m, SHORT[k])]; txt = "%s\\,[%s,\\,%s]" % (pct(r["acc_lenient"]), pct(r["acc_lo"], 0), pct(r["acc_hi"], 0))
            acc.append(("\\textbf{%s}" % txt) if i in best[m] else txt)
        lines.append("%s & %s & %s \\\\" % (SHORT[k], " & ".join(acc), " & ".join("%d / %d" % tuple(wins[m_][k]) for m_ in MODES)))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("t_acc.tex", "\n".join(lines) + "\n")

    # PhD Index, yes-bias, paired consistency
    lines = [FILL % ("l" + "c" * 12), r"\toprule",
             r"& \multicolumn{4}{c}{PhD Index} & \multicolumn{4}{c}{Yes-bias, \%} & \multicolumn{4}{c}{Paired consistency, \%} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}", "Model" + " & Ground & Sway & False & Clash" * 3 + r" \\", r"\midrule"]
    for k in MODELS:
        g = lambda key, f: [f(h[(m, SHORT[k])][key]) for m in mode_names]
        lines.append("%s & %s \\\\" % (SHORT[k], " & ".join(g("phd", lambda v: "%.2f" % v) + g("yes_bias", lambda v: pct(v, 0)) + g("paired_consistency", lambda v: pct(v, 0)))))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("t_other.tex", "\n".join(lines) + "\n")

    # Blank rates and explanation measures
    lines = [FILL % ("l" + "c" * 8), r"\toprule",
             r"& \multicolumn{4}{c}{Blank, \% (Wilson 95\% CI)} & \multicolumn{4}{c}{Read from the explanations} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"Model & MCDR & OEDR & UDR & Control err. & \shortstack{UDR names\\real answer} & \shortstack{Answer vs.\\explanation} & \shortstack{Cites\\caption} & \shortstack{Foreign\\script} \\", r"\midrule"]
    for k in MODELS:
        a, b, c, d = blank[k]; e = ev[k]
        lines.append("%s & %s & %s & %s & %s & %d/30 & %d & %s\\%% & %d \\\\" % (SHORT[k], a[2], b[2], c[2], d[2], e[0], e[1], e[2], e[3]))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("t_blank_all.tex", "\n".join(lines) + "\n")

    # models and cost
    info = {"google/gemini-2.5-flash": ("Google", "off"), "openai/gpt-5-mini": ("OpenAI", "on"), "qwen/qwen2.5-vl-72b-instruct": ("Alibaba", "off"),
            "meta/muse-glimmer-30b": ("Meta", "on"), "mistralai/mistral-large-4-0": ("Mistral AI", "off"), "z-ai/glm-5.3-flash": ("Z.ai", "on")}
    lines = [r"\begin{tabular}{@{}lllrr@{}}", r"\toprule", r"Model & Developer & Reasoning & Calls & Cost (USD) \\", r"\midrule"]
    for k in MODELS:
        n, usd = cost[k][0], cost[k][1]
        lines.append("%s & %s & %s & %s & %.2f \\\\" % (SHORT[k], info[k][0], info[k][1], format(n, ","), usd))
    lines += [r"\midrule", r"Total & & & & %.2f \\" % sum(v[1] for v in cost.values()), r"\bottomrule", r"\end{tabular}"]
    write("t_models.tex", "\n".join(lines) + "\n")

    # ---- numbers used in prose ----
    lines = []
    def mac(name, val):
        lines.append("\\newcommand{\\%s}{%s}" % (name, val))
    mac("nImages", sum(len(imgs[m]) for m in imgs)); mac("nTasksPerModel", sum(len(ref[m]) for m in ref)); mac("nReplies", sum(len(ref[m]) for m in ref) * len(MODELS))
    mac("totalCost", "%.2f" % tot); mac("nOverlapImages", len(overlap)); mac("nPersonImages", len(person))
    write("numbers.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
