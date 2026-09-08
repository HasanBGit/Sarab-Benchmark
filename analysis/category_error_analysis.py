"""
Per-category error-analysis breakdown across Sarab's five evaluation modes,
built entirely from the already-published result CSVs (see the dataset
README's "Result CSVs" section for the schema).

Scope, read this before using: this is a read-only analysis over existing
data, not an ablation study. Every row in the current release has
prompt_version=1 and temperature=0 -- there is no prompt or sampling
variation anywhere in the release to ablate over. Producing a real ablation
(different prompt wording, temperature, reasoning on/off, etc.) means
running new evaluations with modes/*/run_mode*_openrouter.py first; this
script only aggregates what has already been collected.

What this adds beyond each mode script's own `table`/`metrics` commands:
those report one aggregate row per model per mode (the numbers behind the
paper's Table II/III). This script groups the same rows by category
instead, the same way Table IV already does for Clash, and does it for
every mode that has (or can derive) a category:

  base (mode1)  -- direct `category` column, plus a hitem_level (identity/
                   attribute) breakdown. Ground's evaluated 100-task subset
                   is heavily skewed (98/100 rows are "cuisine", 92/100 are
                   "identity"), so this breakdown is printed with an
                   explicit small-sample warning rather than presented as a
                   balanced comparison.
  sec (mode2)   -- direct `category` column. Not currently reported by
                   category anywhere in the paper.
  icc (mode3)   -- direct `category` column. Same as sec.
  ccs (mode4)   -- direct `category` column; reproduces the paper's
                   Table IV as a sanity check that this script's numbers
                   match what is already published.
  nota (mode5)  -- no `category` column; every image_id is
                   "<category>_<slug>", so the category is the first
                   underscore-separated segment. Reported per condition
                   (mcdr/oedr/udr detection rate) and separately as a
                   false-abstention-rate-by-category table, using the same
                   "control rows only, false_abstention = 1 - accuracy on
                   control rows" definition run_mode5_openrouter.py uses.

Usage (run from anywhere -- paths resolve like the mode scripts):
    export SARAB_DATA_DIR=/path/to/Sarab-Dataset-HF   # optional, see below
    python3 analysis/category_error_analysis.py
    python3 analysis/category_error_analysis.py --out-dir analysis/results

Data resolution follows the same convention as modes/*/run_mode*.py:
SARAB_DATA_DIR env override, else a sibling Sarab-Dataset-HF/ checkout next
to this repo, else download+cache the dataset from the Hugging Face Hub
(HassanB4/sarab).

Output: one CSV + one Markdown table per breakdown, under --out-dir
(default analysis/results/), plus a printed summary for all of them.
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _resolve_data_root():
    """Same resolution order as modes/*/run_mode*_openrouter.py: env
    override, sibling checkout, Hugging Face Hub download."""
    override = os.environ.get("SARAB_DATA_DIR")
    if override:
        return Path(override)
    sibling = REPO_ROOT.parent / "Sarab-Dataset-HF"
    if sibling.exists():
        return sibling
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id="HassanB4/sarab", repo_type="dataset"))


DATA_ROOT = _resolve_data_root()

MODEL_LABELS = {
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "qwen/qwen2.5-vl-72b-instruct": "Qwen2.5-VL-72B",
}
MODEL_ORDER = list(MODEL_LABELS.values())


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_all(mode_dir, glob_pattern):
    rows = []
    for path in sorted(mode_dir.glob(glob_pattern)):
        rows.extend(read_rows(path))
    return rows


def model_label(row):
    return MODEL_LABELS.get(row.get("model_name", ""), row.get("model_name", "?"))


def is_correct(row):
    return str(row.get("is_correct")) == "1"


def group_breakdown(rows, group_fn, success_fn=is_correct):
    """{group: {model_label: [successes, total]}}, skipping rows with no
    group (e.g. a blank category column) or a logged error_type -- same
    exclusion every mode script's own compute_metrics() already applies."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.get("error_type"):
            continue
        group = group_fn(r)
        if not group:
            continue
        cell = agg[group][model_label(r)]
        cell[1] += 1
        cell[0] += success_fn(r)
    return agg


def pooled_rate(breakdown, group, models):
    c = t = 0
    for m in models:
        cc, tt = breakdown[group].get(m, [0, 0])
        c += cc
        t += tt
    return (c / t) if t else 0.0


def write_table(breakdown, models, out_csv, out_md, title, hardest_first=True):
    """CSV in long-ish wide form (one row per group, one column per model,
    'correct/total' cells, matching Table IV's own style) plus a Markdown
    table sorted by pooled difficulty."""
    groups = sorted(breakdown.keys(), key=lambda g: pooled_rate(breakdown, g, models), reverse=not hardest_first)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category"] + models + ["pooled_correct", "pooled_total", "pooled_rate"])
        for g in groups:
            row = [g]
            pc = pt = 0
            for m in models:
                c, t = breakdown[g].get(m, [0, 0])
                pc += c
                pt += t
                row.append(f"{c}/{t}" if t else "n/a")
            row += [pc, pt, f"{pc/pt:.3f}" if pt else "n/a"]
            w.writerow(row)

    lines = [f"# {title}", "", "| Category | " + " | ".join(models) + " | Pooled |",
             "|" + "---|" * (len(models) + 2)]
    for g in groups:
        cells = []
        pc = pt = 0
        for m in models:
            c, t = breakdown[g].get(m, [0, 0])
            pc += c
            pt += t
            cells.append(f"{c}/{t}" if t else "n/a")
        pooled = f"{pc}/{pt} ({pc/pt:.0%})" if pt else "n/a"
        lines.append("| " + g + " | " + " | ".join(cells) + f" | {pooled} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(title, breakdown, models):
    print(f"\n=== {title} ===")
    groups = sorted(breakdown.keys(), key=lambda g: pooled_rate(breakdown, g, models), reverse=True)
    for g in groups:
        cells = []
        for m in models:
            c, t = breakdown[g].get(m, [0, 0])
            cells.append(f"{m}: {c}/{t}" if t else f"{m}: n/a")
        print(f"  {g:<32} " + "  ".join(cells))


def analyze(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- mode1 (Ground/base): category + hitem_level, small-sample caveat ----
    m1_dir = DATA_ROOT / "modes" / "mode1_base" / "mode1_results"
    rows1 = read_all(m1_dir, "results_mode1_full_*.csv")
    if rows1:
        by_cat = group_breakdown(rows1, lambda r: r.get("category"))
        by_hitem = group_breakdown(rows1, lambda r: r.get("hitem_level"))
        write_table(by_cat, MODEL_ORDER, out_dir / "base_by_category.csv", out_dir / "base_by_category.md",
                    "Sarab-Ground: accuracy by category")
        write_table(by_hitem, MODEL_ORDER, out_dir / "base_by_hitem_level.csv", out_dir / "base_by_hitem_level.md",
                    "Sarab-Ground: accuracy by task type (identity vs. attribute)")
        print("\n*** Sarab-Ground caveat: the evaluated 100-task subset is 98% 'cuisine' "
              "and 92% 'identity'-type questions (see the paper's Limitations, "
              "\"Ground is evaluated on a subset\"). Treat this breakdown as a hint, "
              "not a balanced per-category/per-task-type comparison. ***")
        print_summary("Sarab-Ground by category", by_cat, MODEL_ORDER)
        print_summary("Sarab-Ground by task type", by_hitem, MODEL_ORDER)

    # ---- mode2 (Sway) and mode3 (False): direct category column ----
    for mode_name, dir_name, results_dir_name, glob, title in [
        ("sec", "mode2_sec", "mode2_results", "results_mode2_sec_*.csv", "Sarab-Sway: accuracy by category"),
        ("icc", "mode3_icc", "mode3_results", "results_mode3_icc_*.csv", "Sarab-False: accuracy by category"),
    ]:
        mode_dir = DATA_ROOT / "modes" / dir_name / results_dir_name
        rows = read_all(mode_dir, glob)
        if not rows:
            continue
        by_cat = group_breakdown(rows, lambda r: r.get("category"))
        write_table(by_cat, MODEL_ORDER, out_dir / f"{mode_name}_by_category.csv",
                    out_dir / f"{mode_name}_by_category.md", title)
        print_summary(title, by_cat, MODEL_ORDER)

    # ---- mode4 (Clash): direct category column (reproduces Table IV) ----
    m4_dir = DATA_ROOT / "modes" / "mode4_ccs" / "mode4_results"
    rows4 = read_all(m4_dir, "results_mode4_ccs_*.csv")
    if rows4:
        by_cat = group_breakdown(rows4, lambda r: r.get("category"))
        write_table(by_cat, MODEL_ORDER, out_dir / "ccs_by_category.csv", out_dir / "ccs_by_category.md",
                    "Sarab-Clash: accuracy by category (reproduces the paper's Table IV)")
        print_summary("Sarab-Clash by category (should match paper Table IV)", by_cat, MODEL_ORDER)

    # ---- mode5 (Blank): category derived from image_id prefix ----
    m5_dir = DATA_ROOT / "modes" / "mode5_nota" / "mode5_results"
    rows5 = read_all(m5_dir, "results_mode5_*.csv")
    if rows5:
        def cat_of(r):
            return r["image_id"].split("_")[0] if r.get("image_id") else None

        core = [r for r in rows5 if str(r.get("is_control")) != "1"]
        control = [r for r in rows5 if str(r.get("is_control")) == "1" and r.get("condition") == "mcdr"]

        for cond in ("mcdr", "oedr", "udr"):
            subset = [r for r in core if r.get("condition") == cond]
            by_cat = group_breakdown(subset, cat_of)
            title = f"Sarab-Blank {cond.upper()}: detection rate by category"
            write_table(by_cat, MODEL_ORDER, out_dir / f"nota_{cond}_by_category.csv",
                        out_dir / f"nota_{cond}_by_category.md", title)
            print_summary(title, by_cat, MODEL_ORDER)

        # false_abstention = 1 - accuracy on control rows, same definition
        # run_mode5_openrouter.py's compute_metrics() uses; here as a
        # per-category rate, so "success" for this table means the model
        # WAS falsely abstaining (wrong on a control item).
        by_cat_fa = group_breakdown(control, cat_of, success_fn=lambda r: not is_correct(r))
        title = "Sarab-Blank: false-abstention rate by category (control items)"
        write_table(by_cat_fa, MODEL_ORDER, out_dir / "nota_false_abstention_by_category.csv",
                    out_dir / "nota_false_abstention_by_category.md", title)
        print_summary(title, by_cat_fa, MODEL_ORDER)

    print(f"\nAll tables written under {out_dir}/")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(HERE / "results"), help="output directory (default: analysis/results/)")
    args = p.parse_args()
    analyze(Path(args.out_dir))


if __name__ == "__main__":
    main()
