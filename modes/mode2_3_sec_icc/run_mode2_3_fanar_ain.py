"""
run_mode2_3_fanar_ain.py
=======================================================================
Sarab Mode 2 (sec) + Mode 3 (icc) evaluation pipeline for Fanar and
AIN, run together as one combined pass (same question format, same
yes/no protocol — splitting them into two separate runs would just
duplicate everything below).

PER-ITEM PROTOCOL
------------------------------------------------------------------------
75 sec items + 75 icc items = 150 items, each with a yes-polarity and
a no-polarity question (question_yes_ar/answer_yes,
question_no_ar/answer_no — identical schema in both files). One turn
per polarity, forced to a single نعم/لا word, auto-scored. 150 items x
2 polarities = 300 calls per model. Every result row carries
"source_mode": "sec" or "icc" so the two can still be analyzed apart.

DATA SOURCE
------------------------------------------------------------------------
modes/mode2_sec/sarab_test_set_sec.json and
modes/mode3_icc/sarab_test_set_icc.json from Sarab-MLLMs/sarab. Images
live under data/candidate_pool/images/<category>/<filename>, fetched
lazily and cached.

ERROR HANDLING / BATCHING
------------------------------------------------------------------------
Same contract as every other mode driver (see fanar_ain_common.py):
append-after-every-row local persistence, error rows retried not
replayed, BATCH_SIZE (50) items at a time per model with a Hub push
after every batch.

RECOVERING FROM A DISCONNECT: drop a previously-downloaded
sarab_sec_icc_results_fanar_ain.json into WORKDIR before running.

CREDENTIALS: HF_TOKEN (write access to Sarab-MLLMs/sarab).

RUNNING IN COLAB: no GitHub clone and nothing to save locally — this
script and its one dependency (fanar_ain_common.py) both live on the
Sarab-MLLMs/sarab dataset repo (public, no token needed just to read
them). One cell:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Sarab-MLLMs/sarab", repo_type="dataset",
                         filename="modes/mode2_3_sec_icc/run_mode2_3_fanar_ain.py",
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
    get_secret, is_error_row, load_item_image, load_results,
    login_hf, now_iso, parse_yes_no, push_results_to_hub, run_in_batches,
    safe_generate_turn, unload_model,
)

# =========================================================================
# Configuration
# =========================================================================

HF_DATASET_REPO = "Sarab-MLLMs/sarab"
SEC_FILENAME = "modes/mode2_sec/sarab_test_set_sec.json"
ICC_FILENAME = "modes/mode3_icc/sarab_test_set_icc.json"
RESULTS_FILENAME = "sarab_sec_icc_results_fanar_ain.json"
RESULTS_REPO_PATH = f"modes/mode2_3_sec_icc/{RESULTS_FILENAME}"

WORKDIR = Path("/content/sarab_sec_icc_eval")
RESULTS_PATH = WORKDIR / RESULTS_FILENAME

POLARITIES = ["yes", "no"]
BATCH_SIZE = 50
MAX_NEW_TOKENS = 64

HF_TOKEN = get_secret("HF_TOKEN")
login_hf(HF_TOKEN)


# =========================================================================
# Per-item inference logic
# =========================================================================

def already_answered(model_name: str, source_mode: str, image_id: str) -> set:
    return {
        r["polarity"] for r in load_results(RESULTS_PATH)
        if r["model_name"] == model_name and r["source_mode"] == source_mode
        and r["image_id"] == image_id and not is_error_row(r)
    }


def run_one_item(handle, item: dict) -> None:
    source_mode = item["mode"]  # "sec" or "icc", already on every live row
    done = already_answered(handle.name, source_mode, item["image_id"])
    if len(done) == len(POLARITIES):
        return
    image = load_item_image(item, HF_DATASET_REPO, WORKDIR)
    if image is None:
        print(f"  [WARN] no image found for {source_mode}/{item['image_id']}, skipping")
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
            "image_id": item["image_id"], "source_mode": source_mode,
            "category": item.get("category"),
            "model_name": handle.name, "polarity": polarity,
            "question_text": question_text, "raw_response": raw_response,
            "parsed_answer": parsed, "ground_truth": ground_truth,
            "is_correct": is_correct, "timestamp": now_iso(),
        }, RESULTS_PATH, key_fields=("model_name", "source_mode", "image_id", "polarity"))


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
                    commit_message=f"Mode2+3 sec/icc inference — {name} — {msg}",
                ),
                label=f"Mode2+3 {spec.name}",
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
    by_key: dict = {}
    for r in results:
        by_key.setdefault((r["model_name"], r["source_mode"]), []).append(r)
    for (model_name, source_mode), rows in sorted(by_key.items()):
        scored = [r for r in rows if not is_error_row(r) and r["is_correct"] is not None]
        acc = sum(r["is_correct"] for r in scored) / len(scored) if scored else 0.0
        print(f"{model_name} / {source_mode}: yes/no accuracy {acc:.1%}  (n={len(scored)}/{len(rows)})")


# =========================================================================
# MAIN — run top-to-bottom in a Colab GPU runtime
# =========================================================================

if __name__ == "__main__":
    sec_items = download_questions_json(HF_DATASET_REPO, SEC_FILENAME, WORKDIR)
    icc_items = download_questions_json(HF_DATASET_REPO, ICC_FILENAME, WORKDIR)
    items = sec_items + icc_items
    download_existing_results_if_any(HF_DATASET_REPO, RESULTS_REPO_PATH, RESULTS_PATH, WORKDIR)
    run_all_inference(items)
    print_metrics()
    print("\nDone. Final results pushed to:",
          f"https://huggingface.co/datasets/{HF_DATASET_REPO}/blob/main/{RESULTS_REPO_PATH}")
