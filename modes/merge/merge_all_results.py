"""
merge_all_results.py
=======================================================================
Sarab benchmark — final merge across all 5 evaluation modes (mode1_base,
mode2_sec, mode3_icc, mode4_ccs, mode5_nota) for Fanar and AIN.

This script does NOT run any model inference. It only reads each
mode's already-finished results file from Sarab-MLLMs/sarab (dataset
repo, public read), normalizes every row into one shared schema, and
writes a single combined file back to the Hub plus a printed accuracy
summary. It has no GPU/Colab dependency — it runs anywhere with
huggingface_hub installed and (for the push step) a write-access HF
token.

UNIFIED ROW SCHEMA
------------------------------------------------------------------------
Every output row has these shared keys:
  mode           "mode1_base" | "mode2_sec" | "mode3_icc" |
                 "mode4_ccs" | "mode5_nota"
  image_id, category, model_name
  polarity       "yes"/"no" for base/sec/icc/ccs, None for nota
  condition      "mcdr"/"oedr"/"udr" for nota, None otherwise
  turn           "initial"/"followup" for ccs, None otherwise
  question_text, raw_response, parsed_answer, ground_truth
  is_correct     0/1, or None if the row errored or hasn't been graded
  is_error       bool — True if raw_response is a captured generation
                 error (e.g. CUDA OOM), same convention as
                 modes/_common/fanar_ain_common.py's is_error_row()
  timestamp
  extra          dict of everything else specific to that mode
                 (followup_judgment, judge_model, ground_truth_ar/en,
                 nota_judgment, nota_position, distractors_ar, ...) —
                 kept so no information is lost, just not promoted to
                 a shared column since it doesn't apply everywhere.

SOURCE FILES (all under Sarab-MLLMs/sarab, repo_type="dataset")
------------------------------------------------------------------------
  modes/mode1_base/sarab_base_results_fanar_ain.json
  modes/mode2_3_sec_icc/sarab_sec_icc_results_fanar_ain.json
    (split into mode2_sec / mode3_icc via its own "source_mode" field)
  modes/mode4_ccs/sarab_ccs_results_fanar_ain.json
  modes/mode5_nota/sarab_nota_results_fanar_ain.json

OUTPUT
------------------------------------------------------------------------
  modes/merge/sarab_all_modes_results_fanar_ain.json  (pushed to Hub)
  Printed per-mode / per-model accuracy summary (stdout). mode5_nota is
  reported per-condition (mcdr/oedr/udr), never pooled into one number,
  since the three conditions test different things and mcdr carries a
  known, accepted, un-rerun confound (its NOTA option is always listed
  last among the choices — see modes/mode5_nota's run script docstring).
"""

import json
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, HfApi

HF_DATASET_REPO = "Sarab-MLLMs/sarab"
WORKDIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = "sarab_all_modes_results_fanar_ain.json"
OUTPUT_REPO_PATH = f"modes/merge/{OUTPUT_FILENAME}"

SOURCE_FILES = {
    "mode1_base": "modes/mode1_base/sarab_base_results_fanar_ain.json",
    "mode2_3_sec_icc": "modes/mode2_3_sec_icc/sarab_sec_icc_results_fanar_ain.json",
    "mode4_ccs": "modes/mode4_ccs/sarab_ccs_results_fanar_ain.json",
    "mode5_nota": "modes/mode5_nota/sarab_nota_results_fanar_ain.json",
}

SHARED_KEYS = (
    "image_id", "category", "model_name", "polarity", "condition", "turn",
    "question_text", "raw_response", "parsed_answer", "ground_truth",
    "is_correct", "timestamp",
)


# =========================================================================
# Download + per-mode normalization
# =========================================================================

def download_source(filename: str) -> list[dict]:
    path = hf_hub_download(repo_id=HF_DATASET_REPO, repo_type="dataset", filename=filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_error_row(row: dict) -> bool:
    return "GENERATION ERROR" in (row.get("raw_response") or "")


def normalize_row(mode: str, row: dict) -> dict:
    out = {"mode": mode}
    for k in SHARED_KEYS:
        out[k] = row.get(k)
    # mode5 uses parsed_letter instead of parsed_answer
    if out["parsed_answer"] is None and "parsed_letter" in row:
        out["parsed_answer"] = row.get("parsed_letter")
    out["is_error"] = is_error_row(row)
    out["extra"] = {k: v for k, v in row.items() if k not in SHARED_KEYS and k != "source_mode"}
    return out


def normalize_mode1(rows: list[dict]) -> list[dict]:
    return [normalize_row("mode1_base", r) for r in rows]


def normalize_mode2_3(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        source = r.get("source_mode")  # "sec" or "icc"
        mode = "mode2_sec" if source == "sec" else "mode3_icc"
        out.append(normalize_row(mode, r))
    return out


def normalize_mode4(rows: list[dict]) -> list[dict]:
    return [normalize_row("mode4_ccs", r) for r in rows]


def normalize_mode5(rows: list[dict]) -> list[dict]:
    return [normalize_row("mode5_nota", r) for r in rows]


# =========================================================================
# Accuracy summary
# =========================================================================

def summarize(all_rows: list[dict]) -> str:
    lines = []
    modes = sorted(set(r["mode"] for r in all_rows))
    for mode in modes:
        mode_rows = [r for r in all_rows if r["mode"] == mode]
        models = sorted(set(r["model_name"] for r in mode_rows))
        lines.append(f"\n=== {mode} ===")
        for model in models:
            rows = [r for r in mode_rows if r["model_name"] == model]
            n_total = len(rows)
            n_err = sum(1 for r in rows if r["is_error"])
            graded = [r for r in rows if not r["is_error"] and r["is_correct"] is not None]
            if mode == "mode4_ccs":
                initial = [r for r in graded if r["turn"] == "initial"]
                acc = sum(r["is_correct"] for r in initial) / len(initial) if initial else 0.0
                followup = [r for r in rows if not r["is_error"] and r["turn"] == "followup"
                            and isinstance(r["extra"].get("followup_judgment"), dict)
                            and r["extra"]["followup_judgment"].get("overall") is not None]
                n_fu = len(followup)
                counts = {"correct": 0, "partial": 0, "incorrect": 0}
                for r in followup:
                    counts[r["extra"]["followup_judgment"]["overall"]] += 1
                line = f"  {model}: n={n_total}, errors={n_err}, turn-1 acc={acc:.1%} (n={len(initial)})"
                if n_fu:
                    line += (f", turn-2 correct={counts['correct']/n_fu:.1%} | "
                             f"partial={counts['partial']/n_fu:.1%} | "
                             f"incorrect={counts['incorrect']/n_fu:.1%} (n={n_fu})")
                lines.append(line)
            elif mode == "mode5_nota":
                # mcdr/oedr/udr test different things (and mcdr's NOTA
                # option is always listed last — a known, accepted,
                # un-rerun positional confound) so a single pooled
                # accuracy across conditions would misrepresent the
                # result. Report each condition separately instead.
                lines.append(f"  {model}: n={n_total}, errors={n_err}")
                for condition in ("mcdr", "oedr", "udr"):
                    cond_graded = [r for r in graded if r["condition"] == condition]
                    cond_acc = (sum(r["is_correct"] for r in cond_graded) / len(cond_graded)
                                if cond_graded else 0.0)
                    note = " (NOTA position never randomized — see limitation)" if condition == "mcdr" else ""
                    lines.append(f"    {condition}: accuracy={cond_acc:.1%} (n={len(cond_graded)}){note}")
            else:
                acc = sum(r["is_correct"] for r in graded) / len(graded) if graded else 0.0
                lines.append(f"  {model}: n={n_total}, errors={n_err}, "
                             f"accuracy={acc:.1%} (n={len(graded)})")
    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    all_rows: list[dict] = []
    all_rows += normalize_mode1(download_source(SOURCE_FILES["mode1_base"]))
    all_rows += normalize_mode2_3(download_source(SOURCE_FILES["mode2_3_sec_icc"]))
    all_rows += normalize_mode4(download_source(SOURCE_FILES["mode4_ccs"]))
    all_rows += normalize_mode5(download_source(SOURCE_FILES["mode5_nota"]))

    out_path = WORKDIR / OUTPUT_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_rows)} merged rows to {out_path}")
    print(summarize(all_rows))

    hf_token = None
    try:
        from google.colab import userdata  # type: ignore
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        import os
        hf_token = os.environ.get("HF_TOKEN")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=OUTPUT_REPO_PATH,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=hf_token,
        commit_message="Merge all 5 modes' final results into one unified file",
    )
    print(f"\nPushed to {HF_DATASET_REPO}:{OUTPUT_REPO_PATH}")


if __name__ == "__main__":
    main()
