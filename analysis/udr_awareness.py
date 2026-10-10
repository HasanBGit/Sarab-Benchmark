"""
UDR "knows but complies" analysis (read-only; uses saved Blank results, no API calls).

UDR shows only wrong options and asks for a letter, so a format-following model can never
score a detection: UDR ~ 0% is a property of the condition. This script measures something
the explanations make visible: how often a model picks a wrong letter while its own
explanation names the REAL, unlisted answer.

Step 1 (automatic): candidates = UDR replies whose explanation contains >= 50% of the
distinctive tokens of the real answer (tokens not in any listed option; the phrase
'المملكة العربية السعودية' is removed first so it cannot match 'برج المملكة').
Step 2 (manual): each candidate was read and labelled TRUE/FALSE; labels are stored in
analysis/udr_awareness_labels.csv. Unlabelled candidates are reported as 'unreviewed'.
The rule alone has about 60% precision (15 of 24 candidates were real), so the labels matter.
Recall is unknown: explanations that name the real answer by an alias are missed, so the
counts are a lower bound.

    python3 analysis/udr_awareness.py
"""
import csv, glob, importlib.util, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = Path(__file__).resolve().parent / "udr_awareness_labels.csv"
GEO = re.compile(r"المملكة العربية السعودية|المملكة العربية|السعودية|المملكة المتحدة")

spec = importlib.util.spec_from_file_location("m5", ROOT / "modes/mode5_nota/run_mode5_openrouter.py")
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

mcdr = {r["image_id"]: r for r in m5.load_records() if r["condition"] == "mcdr" and not r["is_control"]}
labels = {}
if LABELS.is_file():
    for r in csv.DictReader(open(LABELS, encoding="utf-8")):
        labels[r["task_id"] + "|" + r["model"]] = r["label"]

n_rows, rows_out = Counter(), []
for f in sorted(glob.glob(str(ROOT / "modes/mode5_nota/mode5_results/results_mode5_*.csv"))):
    last = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["condition"] == "udr":
            last[r["task_id"]] = r
    for r in last.values():
        it = mcdr[r["image_id"]]
        n_rows[r["model_name"]] += 1
        real_tokens = m5._name_tokens(it["removed_answer_ar"]) - m5._name_tokens(" ".join(it["distractors_ar"]))
        text_tokens = m5._name_tokens(GEO.sub(" ", r["raw_response"]))
        if real_tokens and len(real_tokens & text_tokens) / len(real_tokens) >= 0.5:
            opts = dict(x.split("=", 1) for x in r["options_shown"].split(" | ") if "=" in x)
            rows_out.append({"task_id": r["task_id"], "model": r["model_name"], "letter": r["parsed_letter"],
                             "chosen_option": opts.get(r["parsed_letter"], ""), "real_answer": it["removed_answer_ar"],
                             "label": labels.get(r["task_id"] + "|" + r["model_name"], "unreviewed"),
                             "reply": r["raw_response"].replace("\n", " ")[:300]})

print("%-30s %5s %10s %12s %12s" % ("model", "UDR n", "candidates", "confirmed", "unreviewed"))
for model in sorted(n_rows):
    mine = [r for r in rows_out if r["model"] == model]
    ok = sum(r["label"] == "TRUE" for r in mine)
    print("%-30s %5d %10d %7d (%2.0f%%) %12d" % (model, n_rows[model], len(mine), ok, 100 * ok / n_rows[model],
                                                 sum(r["label"] == "unreviewed" for r in mine)))
if "--write" in sys.argv:
    out = Path(__file__).resolve().parent / "udr_awareness_candidates.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print("wrote", out)
