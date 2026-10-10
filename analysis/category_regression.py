"""
Category breakdown and logistic regression for the full Sarab run. Read-only: saved result CSVs, no API calls.

    python3 analysis/category_regression.py     # writes analysis/results/category_regression.md and category_accuracy.csv

For Ground, Sway and False it fits   correct ~ model + category   as a logistic GEE with image-clustered standard
errors (a yes and a no question share one image, so tasks are not independent). 'correct' uses the lenient parse and
counts a declined reply as wrong. Odds ratios are relative to Gemini 2.5 Flash and to the largest category.
Clash (6 categories, 2 to 3 images each) and Blank are descriptive only.
"""
import csv, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis as ma                      # reuse loading, lenient parsing, Wilson interval

warnings.filterwarnings("ignore")
OUT = ma.OUT
REF_MODEL = "Gemini 2.5 Flash"


def frame():
    data = ma.load(); rows = []
    for (model, mode), rs in data.items():
        if mode not in ma.MODES:
            continue
        for r in rs:
            pred = ma.lenient_parse(r)[1]
            rows.append({"model": ma.SHORT[model], "mode": ma.MODES[mode], "image_id": r["image_id"],
                         "category": r.get("category", ""), "hitem_level": r.get("hitem_level", ""),
                         "ok": ma.is_ok(r["ground_truth"], pred)})
    return pd.DataFrame(rows)


def cell(k, n):
    return "%d/%d (%.0f%%)" % (k, n, 100 * k / n) if n else "-"


def main():
    df = frame(); md = []; csv_rows = []
    models = [ma.SHORT[m] for m in ma.MODEL_ORDER]

    md.append("## 1. Accuracy by category (tasks right / tasks, lenient parse)\n")
    for mode in ma.MODES.values():
        sub = df[df["mode"] == mode]
        cats = sub.groupby("category")["image_id"].nunique().sort_values(ascending=False)
        md.append("\n### %s\n" % mode)
        md.append("| Category | Images | " + " | ".join(models) + " | All models |"); md.append("|" + "---|" * (len(models) + 3))
        for cat in cats.index:
            c = sub[sub["category"] == cat]; cells = []
            for m in models:
                x = c[c["model"] == m]; cells.append(cell(int(x["ok"].sum()), len(x)))
                csv_rows.append([mode, cat, m, int(x["ok"].sum()), len(x)])
            md.append("| %s | %d | %s | %s |" % (cat, cats[cat], " | ".join(cells), cell(int(c["ok"].sum()), len(c))))
        if mode == "Ground":
            md.append("\nGround by question level (identity = naming the item; attribute = a detail of it):\n")
            md.append("| Level | " + " | ".join(models) + " |"); md.append("|" + "---|" * (len(models) + 1))
            for lvl in ("identity", "attribute"):
                c = sub[sub["hitem_level"] == lvl]
                md.append("| %s | %s |" % (lvl, " | ".join(cell(int(c[c["model"] == m]["ok"].sum()), len(c[c["model"] == m])) for m in models)))

    md.append("\n## 2. Logistic regression: correct ~ model + category (GEE, clustered by image)\n")
    md.append("Odds ratio > 1 means more likely right than the reference. Reference model: %s; reference category: the largest one in that mode. "
              "95%% CIs and p-values use image-clustered standard errors.\n" % REF_MODEL)
    for mode in ("Ground", "Sway", "False"):
        sub = df[df["mode"] == mode].copy()
        ref_cat = sub.groupby("category")["image_id"].nunique().idxmax()
        f = "ok ~ C(model, Treatment('%s')) + C(category, Treatment('%s'))" % (REF_MODEL, ref_cat)
        try:
            res = smf.gee(f, groups="image_id", data=sub, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
        except Exception as e:
            md.append("\n### %s\nmodel did not converge: %s\n" % (mode, e)); continue
        or_ = np.exp(res.params); ci = np.exp(res.conf_int())
        md.append("\n### %s (reference category: %s; %d tasks, %d images)\n" % (mode, ref_cat, len(sub), sub["image_id"].nunique()))
        md.append("| Term | Odds ratio | 95% CI | p |"); md.append("|---|---|---|---|")
        for name in res.params.index:
            if name == "Intercept":
                continue
            label = name.replace("C(model, Treatment('%s'))[T." % REF_MODEL, "model: ").replace("C(category, Treatment('%s'))[T." % ref_cat, "category: ").rstrip("]")
            md.append("| %s | %.2f | [%.2f, %.2f] | %s |" % (label, or_[name], ci.loc[name, 0], ci.loc[name, 1],
                                                           "<0.001" if res.pvalues[name] < 0.001 else "%.3f" % res.pvalues[name]))
        # joint Wald tests: does the model matter at all? does the category matter at all?
        try:
            wt = res.wald_test_terms(scalar=True).table
            for term in wt.index:
                if term.startswith("C("):
                    nm = "model" if "model" in term else "category"
                    md.append("\nJoint Wald test, %s: chi2 = %.1f, p %s" % (nm, wt.loc[term].iloc[0], "< 0.001" if wt.loc[term].iloc[1] < 0.001 else "= %.3f" % wt.loc[term].iloc[1]))
        except Exception as e:
            md.append("\n(joint Wald tests unavailable: %s)" % e)
        md.append("")

    md.append("\n## 3. Blank by category (all six models pooled, descriptive)\n")
    data = ma.load(); agg = defaultdict(lambda: [0, 0, 0, 0])
    for (model, mode), rs in data.items():
        if mode != 5:
            continue
        for r in rs:
            cat = r["image_id"].split("_")[0]
            if r["condition"] == "mcdr" and r["is_control"] == "0":
                agg[cat][0] += int(r["is_correct"]); agg[cat][1] += 1
            if r["condition"] == "mcdr" and r["is_control"] == "1":
                agg[cat][2] += 1 - int(r["is_correct"]); agg[cat][3] += 1
    md.append("| Category | MCDR detection | False abstention (control) |"); md.append("|---|---|---|")
    for cat, (a, b, c, d) in sorted(agg.items()):
        lo, hi = ma.wilson(a, b); lo2, hi2 = ma.wilson(c, d)
        md.append("| %s | %s [%.0f, %.0f] | %s [%.0f, %.0f] |" % (cat, cell(a, b), 100 * lo, 100 * hi, cell(c, d), 100 * lo2, 100 * hi2))

    text = "\n".join(md)
    (OUT / "category_regression.md").write_text(text, encoding="utf-8")
    with open(OUT / "category_accuracy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["mode", "category", "model", "tasks_right", "tasks"]); w.writerows(csv_rows)
    print(text)


if __name__ == "__main__":
    main()
