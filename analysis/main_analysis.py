"""
Main analysis of the full Sarab run. Read-only: uses the saved result CSVs, no API calls.

    python3 analysis/main_analysis.py            # prints tables and writes analysis/results/main_*.{md,csv}

What it computes (see research/improvement_plan.md section 4):
  1. strict vs lenient answer parsing (lenient accepts a glued 'لا'/'نعم' at the start of a reply) and a
     separate count of declined/refused replies
  2. accuracy, PhD Index, yes-bias, CATR and paired consistency per model and mode, with 95% CIs from an
     image-clustered bootstrap (a yes and a no question share one image)
  3. exact paired McNemar tests between every pair of models in each mode, Holm-corrected
  4. item difficulty: how many models solve each image, and the images no model solves (for a human label check)
  5. Blank: MCDR / OEDR / UDR / false abstention with Wilson CIs, strict vs exact open-ended scoring,
     and the option-letter distribution (position bias)
"""
import csv, glob, itertools, math, re, importlib.util, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
MODES = {1: "Ground", 2: "Sway", 3: "False", 4: "Clash"}
SHORT = {"google/gemini-2.5-flash": "Gemini 2.5 Flash", "openai/gpt-5-mini": "GPT-5 mini",
         "qwen/qwen2.5-vl-72b-instruct": "Qwen2.5-VL-72B", "meta/muse-glimmer-30b": "Muse Glimmer 30B",
         "mistralai/mistral-large-4-0": "Mistral Large 4", "z-ai/glm-5.3-flash": "GLM-5.3-Flash"}
MODEL_ORDER = list(SHORT)
B, SEED = 2000, 11
YES, NO = "نعم", "لا"
GLUED = re.compile(r"^[\s\"'*#\-–—.,:;!?()\[\]]*(نعم|لا)(?=ال)")
DECLINE = re.compile(r"^[\s\"'*#]*(آسف|عذراً|عذرا)|لا أستطيع (?:تحديد|التأكيد|الجزم)[^.]{0,35}(?:جنسية|أصل|عرق|هوية)")


def latest_rows(path):
    last = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        last[r["task_id"]] = r
    return list(last.values())


def load():
    data = defaultdict(list)                     # (model, mode) -> rows
    for f in sorted(glob.glob(str(ROOT / "modes/mode*/mode*_results/results_*.csv"))):
        mode = int(re.search(r"mode(\d)", f).group(1))
        for r in latest_rows(f):
            data[(r["model_name"], mode)].append(r)
    return data


def lenient_parse(r):
    """yes/no/unclear/declined for a modes 1-4 row; strict = the stored parsed_answer."""
    strict = r["parsed_answer"]
    if r["error_type"] or not r["raw_response"].strip():
        return strict, "error"
    if DECLINE.search(r["raw_response"][:150]):          # a refusal, even if it starts with 'لا'
        return strict, "declined"
    if strict in ("yes", "no"):
        return strict, strict
    g = GLUED.match(r["raw_response"])
    if g:
        return strict, "yes" if g.group(1) == YES else "no"
    return strict, "unclear"


def is_ok(gt, pred):
    return int((pred == "yes" and gt == YES) or (pred == "no" and gt == NO))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def per_image(rows, variant):
    """image_id -> [ok_yes, ok_no, pred_yes_count, n_tasks, declined]"""
    img = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        strict, lenient = lenient_parse(r)
        pred = strict if variant == "strict" else lenient
        a = img[r["image_id"]]
        ok = is_ok(r["ground_truth"], pred)
        if r["ground_truth"] == YES: a[0] += ok
        else: a[1] += ok
        a[2] += pred == "yes"; a[3] += 1; a[4] += (lenient == "declined")
    return img


def metrics_from(arr):
    """arr: array [n_images, 5] -> acc, phd, yes_rate, paired"""
    ok_yes, ok_no, py, n, _ = arr.sum(axis=0)
    ny = len(arr)                               # one yes and one no question per image
    yr, nr = ok_yes / ny, ok_no / ny
    acc = (ok_yes + ok_no) / n
    phd = 2 * yr * nr / (yr + nr) if yr + nr else 0.0
    paired = float(((arr[:, 0] + arr[:, 1]) == 2).mean())
    return acc, phd, py / n, paired


def boot_ci(arr, rng):
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    sums = arr[idx].sum(axis=1)                 # [B, 5]
    ny = len(arr)
    yr, nr = sums[:, 0] / ny, sums[:, 1] / ny
    acc = (sums[:, 0] + sums[:, 1]) / sums[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        phd = np.where(yr + nr > 0, 2 * yr * nr / (yr + nr), 0.0)
    return {"acc": np.percentile(acc, [2.5, 97.5]), "phd": np.percentile(phd, [2.5, 97.5])}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def main():
    data = load()
    rng = np.random.default_rng(SEED)
    md = []

    # ---------------- 1+2: headline metrics, strict vs lenient, with CIs ----------------
    md.append("## 1. Headline metrics (95%% CI from an image-clustered bootstrap, %d resamples)\n" % B)
    md.append("Lenient parsing also accepts a glued 'لا'/'نعم' at the start of a reply; declined replies stay wrong.\n")
    rows_csv, declines = [], []
    for mode, name in MODES.items():
        md.append("\n### %s\n" % name)
        hdr = "| Model | Acc strict | Acc lenient (95%% CI) | PhD lenient (95%% CI) | Yes-bias | %s | Paired | Declined | Unparsed (lenient) | Rescued |" % ("CATR" if mode in (2, 3) else "n")
        md.append(hdr); md.append("|" + "---|" * 10)
        for m in MODEL_ORDER:
            rows = data[(m, mode)]
            s_arr = np.array(list(per_image(rows, "strict").values()), dtype=float)
            l_arr = np.array(list(per_image(rows, "lenient").values()), dtype=float)
            acc_s = metrics_from(s_arr)[0]; acc, phd, yb, paired = metrics_from(l_arr)
            ci = boot_ci(l_arr, rng)
            parsed = [lenient_parse(r) for r in rows]
            declined = sum(p[1] == "declined" for p in parsed); unclear = sum(p[1] in ("unclear", "error") for p in parsed)
            rescued = sum(p[0] not in ("yes", "no") and p[1] in ("yes", "no") for p in parsed)
            if declined:
                as_scored = sum(int(r["is_correct"]) for r in rows) / len(rows)
                declines.append("- %s, %s: %d declined replies. Accuracy as scored by the run %.1f%%; with declines counted as wrong %.1f%% (%d of them were scored correct only because they start with 'لا')." % (
                    name, SHORT[m], declined, 100 * as_scored, 100 * acc, sum(1 for r in rows if lenient_parse(r)[1] == "declined" and int(r["is_correct"]))))
            extra = ("%.0f%%" % (100 * (1 - acc))) if mode in (2, 3) else str(len(rows))
            md.append("| %s | %.1f%% | %.1f%% [%.1f, %.1f] | %.3f [%.3f, %.3f] | %.0f%% | %s | %.0f%% | %d | %d | %d |" % (
                SHORT[m], 100 * acc_s, 100 * acc, 100 * ci["acc"][0], 100 * ci["acc"][1], phd, ci["phd"][0], ci["phd"][1],
                100 * yb, extra, 100 * paired, declined, unclear, rescued))
            rows_csv.append([name, SHORT[m], len(rows), round(acc_s, 4), round(acc, 4), round(ci["acc"][0], 4), round(ci["acc"][1], 4),
                             round(phd, 4), round(ci["phd"][0], 4), round(ci["phd"][1], 4), round(yb, 4), round(paired, 4), declined, unclear, rescued])
    md.append("\n### Declined replies (refusals to infer a person's nationality or origin)\n")
    md.extend(declines or ["- none"])
    md.append("\nIn the headline table, 'Acc lenient' already counts a declined reply as wrong.\n")
    with open(OUT / "main_headline.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "model", "n_tasks", "acc_strict", "acc_lenient", "acc_lo", "acc_hi", "phd", "phd_lo", "phd_hi",
                    "yes_bias", "paired_consistency", "declined", "unparsed", "rescued_by_lenient"])
        w.writerows(rows_csv)

    # ---------------- 3: paired McNemar ----------------
    md.append("\n## 2. Paired significance (exact McNemar on the same tasks, Holm-corrected within each mode)\n")
    md.append("Only pairs with Holm-adjusted p < 0.05 are listed. 'A>B' = A right and B wrong more often than the reverse.\n")
    mc_csv = []
    for mode, name in MODES.items():
        pairs, ps, info = [], [], []
        by = {}
        for m in MODEL_ORDER:
            by[m] = {r["task_id"]: is_ok(r["ground_truth"], lenient_parse(r)[1]) for r in data[(m, mode)]}
        for a, b in itertools.combinations(MODEL_ORDER, 2):
            common = set(by[a]) & set(by[b])
            ab = sum(by[a][t] and not by[b][t] for t in common); ba = sum(by[b][t] and not by[a][t] for t in common)
            p = 1.0 if ab + ba == 0 else binomtest(ab, ab + ba, 0.5).pvalue
            pairs.append((a, b)); ps.append(p); info.append((ab, ba))
        adj = holm(np.array(ps))
        sig = [(a, b, i, p, q) for (a, b), i, p, q in zip(pairs, info, ps, adj) if q < 0.05]
        md.append("- **%s**: %d of %d pairs significant." % (name, len(sig), len(pairs)))
        for a, b, (ab, ba), p, q in sorted(sig, key=lambda x: x[4]):
            win, lose, w1, w2 = (a, b, ab, ba) if ab > ba else (b, a, ba, ab)
            md.append("    - %s > %s (%d vs %d tasks, p=%.4f, Holm p=%.4f)" % (SHORT[win], SHORT[lose], w1, w2, p, q))
        for (a, b), (ab, ba), p, q in zip(pairs, info, ps, adj):
            mc_csv.append([name, SHORT[a], SHORT[b], ab, ba, round(p, 6), round(float(q), 6)])
    with open(OUT / "main_mcnemar.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["mode", "model_a", "model_b", "a_only_right", "b_only_right", "p", "holm_p"]); w.writerows(mc_csv)

    # ---------------- 4: item difficulty ----------------
    md.append("\n## 3. Item difficulty: images solved (both questions right) by k of 6 models\n")
    md.append("| Mode | Images | 0 models | 1 | 2 | 3 | 4 | 5 | 6 | None solve | All solve |"); md.append("|" + "---|" * 11)
    fail_rows = []
    for mode, name in MODES.items():
        solved = defaultdict(int); meta = {}
        for m in MODEL_ORDER:
            for img, a in per_image(data[(m, mode)], "lenient").items():
                solved[img] += int(a[0] + a[1] == 2)
        for r in data[(MODEL_ORDER[0], mode)]:
            meta.setdefault(r["image_id"], {})[r["polarity"]] = r["question_text"]; meta[r["image_id"]]["cat"] = r.get("category", "")
        dist = Counter(solved.values()); n = len(solved)
        md.append("| %s | %d | %s | %.0f%% | %.0f%% |" % (name, n, " | ".join(str(dist.get(k, 0)) for k in range(7)),
                                                         100 * dist.get(0, 0) / n, 100 * dist.get(6, 0) / n))
        for img, k in solved.items():
            if k == 0:
                fail_rows.append([name, img, meta[img].get("cat", ""), meta[img].get("yes", ""), meta[img].get("no", "")])
    with open(OUT / "main_all_fail_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["mode", "image_id", "category", "yes_question", "no_question"]); w.writerows(fail_rows)
    md.append("\nImages no model solves are listed in `analysis/results/main_all_fail_items.csv` (%d rows) for a native-speaker label check.\n" % len(fail_rows))

    # ---------------- 5: Blank ----------------
    spec = importlib.util.spec_from_file_location("m5", ROOT / "modes/mode5_nota/run_mode5_openrouter.py")
    m5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m5)
    dis = {r["image_id"]: r["distractors_ar"] for r in m5.load_records() if r["condition"] == "mcdr" and not r["is_control"]}
    md.append("\n## 4. Blank (absent-answer detection), 30 items per condition, Wilson 95% CI\n")
    md.append("| Model | MCDR | OEDR (scored) | OEDR exact-only | UDR | False abstention | Unanswered |"); md.append("|" + "---|" * 7)
    letters = {}
    for m in MODEL_ORDER:
        rows = data[(m, 5)]
        def cond(c, control=False):
            return [r for r in rows if r["condition"] == c and (r["is_control"] == "1") == control]
        def cell(rs, f):
            k = sum(f(r) for r in rs); n = len(rs); lo, hi = wilson(k, n); return "%.0f%% [%.0f, %.0f]" % (100 * k / n, 100 * lo, 100 * hi)
        mcdr, oedr, udr, ctl = cond("mcdr"), cond("oedr"), cond("udr"), cond("mcdr", True)
        exact = lambda r: int(not r["error_type"] and not any(d in m5.oedr_answer(r["raw_response"]) for d in dis[r["image_id"]]))
        fa = lambda r: 1 - int(r["is_correct"])
        unans = sum(1 for r in rows if r["error_type"] or not r["raw_response"].strip())
        md.append("| %s | %s | %s | %s | %s | %s | %d |" % (SHORT[m], cell(mcdr, lambda r: int(r["is_correct"])), cell(oedr, lambda r: int(r["is_correct"])),
                  cell(oedr, exact), cell(udr, lambda r: int(r["is_correct"])), cell(ctl, fa), unans))
        c = Counter(r["parsed_letter"] for r in mcdr + ctl + udr if r["parsed_letter"] in ("A", "B", "C", "D"))
        letters[m] = c
    md.append("\n### Option-letter choices (MCDR + control + UDR, 90 letter tasks per model)\n")
    md.append("| Model | A | B | C | D |"); md.append("|" + "---|" * 5)
    for m in MODEL_ORDER:
        c = letters[m]; md.append("| %s | %d | %d | %d | %d |" % ((SHORT[m],) + tuple(c.get(x, 0) for x in "ABCD")))
    md.append("\nOptions are shuffled per task, so a strong lean to one letter would show a position bias. D exists only in the MCDR and control tasks (60 of the 90), so read the table as descriptive, not as a test.\n")

    text = "\n".join(md)
    (OUT / "main_tables.md").write_text(text, encoding="utf-8")
    print(text)
    print("\nwrote", OUT / "main_tables.md", "and CSVs in", OUT)


if __name__ == "__main__":
    main()
