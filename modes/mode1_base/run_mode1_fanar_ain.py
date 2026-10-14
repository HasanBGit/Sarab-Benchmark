"""
run_mode1_fanar_ain.py
=======================================================================
Sarab Mode 1 (base) evaluation pipeline for Fanar and AIN.

PER-ITEM PROTOCOL
------------------------------------------------------------------------
270 images, each with a yes-polarity and a no-polarity question
(question_yes_ar/answer_yes, question_no_ar/answer_no). One turn per
polarity, forced to a single نعم/لا word, auto-scored against the
ground-truth answer. 270 items x 2 polarities = 540 calls per model.

DATA SOURCE
------------------------------------------------------------------------
modes/mode1_base/questions_base.json (a dict with "items": [...]) from
Sarab-MLLMs/sarab. Each item's images live under
data/candidate_pool/images/<category>/<filename> — fetched lazily and
cached by load_item_image().

ERROR HANDLING / BATCHING
------------------------------------------------------------------------
Processed BATCH_SIZE (50) items at a time per model; results pushed to
the Hub after every batch, not just once per model, so a disconnect
loses at most one batch's unpushed progress (and at most one row of
that batch — see fanar_ain_common.py's append-after-every-row
contract). Resuming re-runs this script: a model with all 540 rows
already present is skipped; a partially-done model picks up mid-item.

RECOVERING FROM A DISCONNECT
------------------------------------------------------------------------
Before running, if you have a previously-downloaded
sarab_base_results_fanar_ain.json, drop it into WORKDIR so resuming
sees it instead of starting from zero (same as the other mode drivers).

CREDENTIALS: HF_TOKEN (write access to Sarab-MLLMs/sarab).

RUNNING IN COLAB: no GitHub clone and nothing to save locally — this
script and its one dependency (fanar_ain_common.py) both live on the
Sarab-MLLMs/sarab dataset repo (public, no token needed just to read
them). One cell:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Sarab-MLLMs/sarab", repo_type="dataset",
                         filename="modes/mode1_base/run_mode1_fanar_ain.py",
                         local_dir="/content")
    !python "{p}"

Author: Zahra Alharz. Pipeline co-written with Claude Code.
"""

# !pip install -q -U transformers accelerate "bitsandbytes>=0.46.1" \
#     qwen-vl-utils huggingface_hub pillow

# =========================================================================
# Setup — fetch fanar_ain_common.py from the Hub and import it
# =========================================================================

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

_COMMON_FILE = hf_hub_download(
    repo_id="Sarab-MLLMs/sarab", repo_type="dataset",
    filename="modes/_common/fanar_ain_common.py", local_dir="/content/_sarab_pipeline_code",
)
sys.path.insert(0, str(Path(_COMMON_FILE).parent))

from fanar_ain_common import (  # noqa: E402
    MODEL_SPECS, SYSTEM_INSTRUCTION_YES_NO_AR,
    append_result_row, download_existing_results_if_any, download_questions_json,
    get_secret, is_error_row, load_item_image, load_results, login_hf,
    safe_generate_turn,
    now_iso, parse_yes_no, push_results_to_hub, run_in_batches, unload_model,
)

# =========================================================================
# Configuration
# =========================================================================

HF_DATASET_REPO = "Sarab-MLLMs/sarab"
QUESTIONS_FILENAME = "modes/mode1_base/questions_base.json"
RESULTS_FILENAME = "sarab_base_results_fanar_ain.json"
RESULTS_REPO_PATH = f"modes/mode1_base/{RESULTS_FILENAME}"

WORKDIR = Path("/content/sarab_base_eval")
RESULTS_PATH = WORKDIR / RESULTS_FILENAME

POLARITIES = ["yes", "no"]
BATCH_SIZE = 50
MAX_NEW_TOKENS = 64

HF_TOKEN = get_secret("HF_TOKEN")
login_hf(HF_TOKEN)


# =========================================================================
# Per-item inference logic
# =========================================================================

def already_answered(model_name: str, image_id: str) -> set:
    return {
        r["polarity"] for r in load_results(RESULTS_PATH)
        if r["model_name"] == model_name and r["image_id"] == image_id and not is_error_row(r)
    }


def run_one_item(handle, item: dict) -> None:
    done = already_answered(handle.name, item["image_id"])
    if len(done) == len(POLARITIES):
        return
    image = load_item_image(item, HF_DATASET_REPO, WORKDIR)
    if image is None:
        print(f"  [WARN] no image found for {item['image_id']}, skipping")
        return

    for polarity in POLARITIES:
        if polarity in done:
            continue
        question_field = "question_yes_ar" if polarity == "yes" else "question_no_ar"
        answer_field = "answer_yes" if polarity == "yes" else "answer_no"
        question_text = item[question_field]
        ground_truth = item[answer_field]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION_YES_NO_AR}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ]},
        ]
        raw_response = safe_generate_turn(handle, messages, MAX_NEW_TOKENS)

        parsed = parse_yes_no(raw_response) if "GENERATION ERROR" not in raw_response else None
        is_correct = None
        if parsed is not None:
            is_correct = int((parsed == "yes" and ground_truth == "نعم") or
                              (parsed == "no" and ground_truth == "لا"))

        append_result_row({
            "image_id": item["image_id"], "category": item.get("category"),
            "model_name": handle.name, "polarity": polarity,
            "question_text": question_text, "raw_response": raw_response,
            "parsed_answer": parsed, "ground_truth": ground_truth,
            "is_correct": is_correct, "timestamp": now_iso(),
        }, RESULTS_PATH, key_fields=("model_name", "image_id", "polarity"))


# =========================================================================
# Orchestration — run every model over every item, batched
# =========================================================================

def run_all_inference(items: list[dict]) -> None:
    expected_rows_per_model = len(items) * len(POLARITIES)
    for spec in MODEL_SPECS:
        results = load_results(RESULTS_PATH)
        rows_done = sum(1 for r in results if r["model_name"] == spec.name and not is_error_row(r))
        if rows_done >= expected_rows_per_model:
            print(f"\n=== Skipping {spec.name} — already has all {expected_rows_per_model} rows ===")
            continue

        print(f"\n=== Loading {spec.name} ({rows_done}/{expected_rows_per_model} rows already done) ===")
        handle = spec.loader()
        try:
            run_in_batches(
                items,
                run_one_item_fn=lambda item, h=handle: run_one_item(h, item),
                push_fn=lambda msg, name=spec.name: push_results_to_hub(
                    HF_TOKEN, RESULTS_PATH, HF_DATASET_REPO, RESULTS_REPO_PATH,
                    commit_message=f"Mode1 base inference — {name} — {msg}",
                ),
                label=f"Mode1 {spec.name}",
                batch_size=BATCH_SIZE,
            )
        finally:
            unload_model(handle)
            print(f"=== Unloaded {spec.name} ===")


# =========================================================================
# Metrics
# =========================================================================

def print_metrics() -> None:
    results = load_results(RESULTS_PATH)
    by_model: dict = {}
    for r in results:
        by_model.setdefault(r["model_name"], []).append(r)
    for model_name, rows in by_model.items():
        scored = [r for r in rows if not is_error_row(r) and r["is_correct"] is not None]
        acc = sum(r["is_correct"] for r in scored) / len(scored) if scored else 0.0
        print(f"{model_name}: yes/no accuracy {acc:.1%}  (n={len(scored)}/{len(rows)})")


# =========================================================================
# MAIN — run top-to-bottom in a Colab GPU runtime
# =========================================================================

if __name__ == "__main__":
    data = download_questions_json(HF_DATASET_REPO, QUESTIONS_FILENAME, WORKDIR)
    items = data["items"]
    download_existing_results_if_any(HF_DATASET_REPO, RESULTS_REPO_PATH, RESULTS_PATH, WORKDIR)
    run_all_inference(items)
    print_metrics()
    print("\nDone. Final results pushed to:",
          f"https://huggingface.co/datasets/{HF_DATASET_REPO}/blob/main/{RESULTS_REPO_PATH}")
