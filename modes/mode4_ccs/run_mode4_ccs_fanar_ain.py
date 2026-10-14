"""
run_mode4_ccs_fanar_ain.py
=======================================================================
Sarab Mode 4 (ccs — Cultural Counter-Common-Sense / Parametric Prior
Hijacking) evaluation pipeline for Fanar and AIN.

RESEARCH QUESTION
------------------------------------------------------------------------
Same hypothesis as the Fatan pipeline: open-weight VLMs marketed as
"culturally aligned"/"Arabic-centric" (Fanar, AIN) are hypothesized to
NOT reliably ground their answers in the actual pixels once a
plausible, culturally-expected premise is in play. Mode 4's 28 images
each bury one real norm violation inside an otherwise
culturally-familiar scene.

PER-ITEM PROTOCOL (2 turns, per polarity, per model)
------------------------------------------------------------------------
Each image has question_yes_ar/question_no_ar, like mode1_base. For
EACH polarity, 2 turns:
  Turn 1 (initial):  forced-choice نعم/لا, auto-scored against
                      answer_yes/answer_no.
  Turn 2 (followup):  followup_question_ar ("...explain, don't just
                      say yes or no"). Open-ended — tests whether the
                      model can actually articulate ground_truth_ar,
                      or whether turn 1 was a lucky/unlucky guess with
                      no real grounding behind it. Graded by a Gemini
                      judge against ground_truth_ar + failure_mechanism
                      + each item's own judge_checklist (per-item
                      Fatan-style checklist, met true/false/rationale),
                      which also gives one overall verdict —
                      correct/partial/incorrect — plus its own
                      rationale (not a mechanical function of the
                      checklist booleans).
28 images x 2 models x 2 polarities x 2 turns = 224 calls total.

DATA SOURCE: modes/mode4_ccs/sarab_ccs_questions.json (28 entries) +
modes/mode4_ccs/images/*.{jpg,png}, Sarab-MLLMs/sarab.

ERROR HANDLING / BATCHING
------------------------------------------------------------------------
Same contract as every other mode driver (see fanar_ain_common.py):
append-after-every-row local persistence, error rows retried not
replayed, BATCH_SIZE (50, i.e. effectively one batch for 28 items)
images at a time per model with a Hub push after every batch.

RECOVERING FROM A DISCONNECT: drop a previously-downloaded
sarab_ccs_results_fanar_ain.json into WORKDIR before running.

CREDENTIALS: HF_TOKEN (write), OPENROUTER_API_KEY (Gemini followup judge).

RUNNING IN COLAB: no GitHub clone and nothing to save locally — this
script and its one dependency (fanar_ain_common.py) both live on the
Sarab-MLLMs/sarab dataset repo (public, no token needed just to read
them). One cell:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Sarab-MLLMs/sarab", repo_type="dataset",
                         filename="modes/mode4_ccs/run_mode4_ccs_fanar_ain.py",
                         local_dir="/content")
    !python "{p}"

Author: Zahra Alharz. Pipeline co-written with Claude Code.
"""

# !pip install -q -U transformers accelerate "bitsandbytes>=0.46.1" \
#     qwen-vl-utils huggingface_hub openai pillow

# =========================================================================
# Setup — fetch fanar_ain_common.py from the Hub and import it
# =========================================================================

import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

_COMMON_FILE = hf_hub_download(
    repo_id="Sarab-MLLMs/sarab", repo_type="dataset",
    filename="modes/_common/fanar_ain_common.py", local_dir="/content/_sarab_pipeline_code",
)
sys.path.insert(0, str(Path(_COMMON_FILE).parent))

from fanar_ain_common import (  # noqa: E402
    GEMINI_JUDGE_MODEL, MODEL_SPECS, SYSTEM_INSTRUCTION_YES_NO_AR,
    append_result_row, build_openrouter_client, download_existing_results_if_any,
    download_questions_json, get_secret, is_error_row, load_item_image,
    load_results, login_hf, now_iso, parse_yes_no, push_results_to_hub,
    run_in_batches, safe_generate_turn, unload_model,
)

# =========================================================================
# Configuration
# =========================================================================

HF_DATASET_REPO = "Sarab-MLLMs/sarab"
QUESTIONS_FILENAME = "modes/mode4_ccs/sarab_ccs_questions.json"
RESULTS_FILENAME = "sarab_ccs_results_fanar_ain.json"
RESULTS_REPO_PATH = f"modes/mode4_ccs/{RESULTS_FILENAME}"

WORKDIR = Path("/content/sarab_ccs_eval")
RESULTS_PATH = WORKDIR / RESULTS_FILENAME

POLARITIES = ["yes", "no"]
TURNS = ["initial", "followup"]
BATCH_SIZE = 50
MAX_NEW_TOKENS_INITIAL = 64
MAX_NEW_TOKENS_FOLLOWUP = 800

HF_TOKEN = get_secret("HF_TOKEN")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
login_hf(HF_TOKEN)


# =========================================================================
# Per-item inference logic
# =========================================================================

def already_answered(model_name: str, image_id: str, polarity: str) -> dict:
    """turn -> raw_response for turns already completed for this
    (model, image, polarity). Error rows excluded so a failed turn is
    retried, not replayed as if it were a real answer."""
    return {
        r["turn"]: r["raw_response"]
        for r in load_results(RESULTS_PATH)
        if r["model_name"] == model_name and r["image_id"] == image_id
        and r["polarity"] == polarity and not is_error_row(r)
    }


def run_item_polarity(handle, item: dict, polarity: str, image) -> None:
    question_field = "question_yes_ar" if polarity == "yes" else "question_no_ar"
    answer_field = "answer_yes" if polarity == "yes" else "answer_no"
    question_text = item[question_field]
    ground_truth = item[answer_field]

    done = already_answered(handle.name, item["image_id"], polarity)
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION_YES_NO_AR}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question_text},
        ]},
    ]

    if "initial" in done:
        initial_response = done["initial"]
    else:
        initial_response = safe_generate_turn(handle, messages, MAX_NEW_TOKENS_INITIAL)
        append_result_row({
            "image_id": item["image_id"], "category": item["category"],
            "model_name": handle.name, "polarity": polarity, "turn": "initial",
            "question_text": question_text, "raw_response": initial_response,
            "timestamp": now_iso(), "ground_truth": ground_truth,
            "parsed_answer": parse_yes_no(initial_response) if "GENERATION ERROR" not in initial_response else None,
            "is_correct": None,
            "ground_truth_ar": item["ground_truth_ar"], "ground_truth_en": item.get("ground_truth_en"),
            "followup_question": None, "followup_judgment": None, "judge_model": None,
        }, RESULTS_PATH, key_fields=("model_name", "image_id", "polarity", "turn"))
    messages.append({"role": "assistant", "content": [{"type": "text", "text": initial_response}]})

    if "GENERATION ERROR" in initial_response:
        return  # don't stack the followup on a failed initial turn

    if "followup" in done:
        return
    followup_text = item["followup_question_ar"]
    messages.append({"role": "user", "content": [{"type": "text", "text": followup_text}]})
    followup_response = safe_generate_turn(handle, messages, MAX_NEW_TOKENS_FOLLOWUP)

    append_result_row({
        "image_id": item["image_id"], "category": item["category"],
        "model_name": handle.name, "polarity": polarity, "turn": "followup",
        "question_text": followup_text, "raw_response": followup_response,
        "timestamp": now_iso(), "ground_truth": None, "parsed_answer": None, "is_correct": None,
        "ground_truth_ar": item["ground_truth_ar"], "ground_truth_en": item.get("ground_truth_en"),
        "failure_mechanism": item.get("failure_mechanism"),
        "judge_checklist": item.get("judge_checklist"),
        "followup_question": followup_text, "followup_judgment": None, "judge_model": None,
    }, RESULTS_PATH, key_fields=("model_name", "image_id", "polarity", "turn"))


def run_one_item(handle, item: dict) -> None:
    image = load_item_image(item, HF_DATASET_REPO, WORKDIR)
    if image is None:
        print(f"  [WARN] no image found for {item['image_id']}, skipping")
        return
    for polarity in POLARITIES:
        run_item_polarity(handle, item, polarity, image)


# =========================================================================
# Orchestration — run every model over every item, batched
# =========================================================================

def run_all_inference(items: list[dict]) -> None:
    expected_rows_per_model = len(items) * len(POLARITIES) * len(TURNS)
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
                    commit_message=f"Mode4 ccs inference — {name} — {msg}",
                ),
                label=f"Mode4 {spec.name}",
                batch_size=BATCH_SIZE,
            )
        finally:
            unload_model(handle)
            print(f"=== Unloaded {spec.name} ===")


# =========================================================================
# Grading
# =========================================================================

def grade_initial_turns() -> None:
    results = load_results(RESULTS_PATH)
    for row in results:
        if row["turn"] != "initial" or is_error_row(row):
            continue
        if row["is_correct"] is not None:
            continue
        parsed, gt = row["parsed_answer"], row["ground_truth"]
        row["is_correct"] = int((parsed == "yes" and gt == "نعم") or (parsed == "no" and gt == "لا"))
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


_FOLLOWUP_JUDGE_PROMPT_TEMPLATE = """You are grading a vision-language model's \
free-text explanation of what it sees in a culturally-themed image, after it \
was first asked a yes/no question about the same image and told not to just \
answer yes or no this time.

You are given: the ground-truth fact about the image, the typical failure \
mode models show on this item (context only — the model doesn't need to name \
the failure mode itself, only avoid falling into it), a checklist of specific \
points a fully correct answer should make, and the model's free-text \
explanation.

Step 1 — grade EACH checklist item independently: does the model's \
explanation make that point (even if worded differently from the checklist \
text)? true or false, with a short one-sentence rationale citing what the \
answer did or did not say.

Step 2 — give an OVERALL verdict for the whole explanation:
  "correct" — substantively captures the ground truth.
  "partial" — gestures at the right idea but is vague, incomplete, or misses \
a key specific the checklist calls for.
  "incorrect" — misses the point entirely, or asserts something false.
Use your own judgment for the overall verdict — it is not required to be a \
strict function of the checklist booleans (e.g. an answer could technically \
hit every checklist item yet still read as confused or hedged enough to \
downgrade, or vice versa) — but justify your call in the rationale.

Ground truth:
{ground_truth}

Known failure mode for this item (context only):
{failure_mechanism}

Checklist (grade each independently):
{checklist_items}

Model's explanation:
{answer}

Respond with ONLY a JSON object shaped exactly like:
{{"checklist": [{{"item": "<checklist item text, copied exactly>", "met": true or false, "rationale": "<one sentence>"}}], "overall": "correct", "overall_rationale": "<1-2 sentences>"}}
No other text before or after the JSON object."""


def grade_followup_gemini(client, ground_truth: str, failure_mechanism: str,
                           checklist: list[str], raw_response: str) -> dict:
    checklist_items = "\n".join(f"- {item}" for item in checklist)
    prompt = _FOLLOWUP_JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth, failure_mechanism=failure_mechanism or "(none given)",
        checklist_items=checklist_items, answer=raw_response,
    )
    response = client.chat.completions.create(
        model=GEMINI_JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        judgment = json.loads(text)
        if (isinstance(judgment, dict) and "overall" in judgment
                and judgment["overall"] in ("correct", "partial", "incorrect")
                and isinstance(judgment.get("checklist"), list)
                and len(judgment["checklist"]) == len(checklist)):
            return judgment
    except json.JSONDecodeError:
        pass
    return {
        "checklist": [{"item": item, "met": None, "rationale": "JUDGE_PARSE_ERROR"} for item in checklist],
        "overall": None,
        "overall_rationale": "JUDGE_PARSE_ERROR — raw judge output: " + text,
    }


def grade_followup_turns() -> None:
    client = build_openrouter_client(OPENROUTER_API_KEY)
    results = load_results(RESULTS_PATH)
    for row in results:
        if row["turn"] != "followup" or is_error_row(row):
            continue
        if row["followup_judgment"] is not None and row["followup_judgment"].get("overall") is not None:
            continue
        row["followup_judgment"] = grade_followup_gemini(
            client, row["ground_truth_ar"], row.get("failure_mechanism"),
            row.get("judge_checklist") or [], row["raw_response"],
        )
        row["judge_model"] = GEMINI_JUDGE_MODEL
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def run_grading() -> None:
    grade_initial_turns()
    grade_followup_turns()
    push_results_to_hub(HF_TOKEN, RESULTS_PATH, HF_DATASET_REPO, RESULTS_REPO_PATH,
                         commit_message="Mode4 ccs grading pass complete (yes/no auto-score + Gemini followup judge)")
    print("Grading complete.")


# =========================================================================
# Metrics
# =========================================================================

def print_metrics() -> None:
    results = load_results(RESULTS_PATH)
    by_model: dict = {}
    for r in results:
        by_model.setdefault(r["model_name"], {"initial": [], "followup": []})
        by_model[r["model_name"]][r["turn"]].append(r)

    for model_name, turns in by_model.items():
        initial = [r for r in turns["initial"] if not is_error_row(r) and r["is_correct"] is not None]
        followup = [r for r in turns["followup"] if not is_error_row(r)
                    and r["followup_judgment"] and r["followup_judgment"].get("overall") is not None]
        acc = sum(r["is_correct"] for r in initial) / len(initial) if initial else 0.0
        n_fu = len(followup)
        counts = {"correct": 0, "partial": 0, "incorrect": 0}
        for r in followup:
            counts[r["followup_judgment"]["overall"]] += 1
        print(f"\n=== {model_name} ===")
        print(f"turn-1 yes/no accuracy: {acc:.1%}  (n={len(initial)})")
        if n_fu:
            print(f"turn-2 explanation verdict (n={n_fu}): "
                  f"correct {counts['correct']/n_fu:.1%} | "
                  f"partial {counts['partial']/n_fu:.1%} | "
                  f"incorrect {counts['incorrect']/n_fu:.1%}")
        else:
            print("turn-2 explanation verdict: n=0")
        print("  (high turn-1 / low turn-2-correct means the yes/no answer was "
              "a lucky guess, not real grounding.)")


# =========================================================================
# MAIN — run top-to-bottom in a Colab GPU runtime
# =========================================================================

if __name__ == "__main__":
    items = download_questions_json(HF_DATASET_REPO, QUESTIONS_FILENAME, WORKDIR)
    download_existing_results_if_any(HF_DATASET_REPO, RESULTS_REPO_PATH, RESULTS_PATH, WORKDIR)
    run_all_inference(items)
    run_grading()
    print_metrics()
    print("\nDone. Final results pushed to:",
          f"https://huggingface.co/datasets/{HF_DATASET_REPO}/blob/main/{RESULTS_REPO_PATH}")
