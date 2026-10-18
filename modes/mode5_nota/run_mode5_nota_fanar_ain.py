"""
run_mode5_nota_fanar_ain.py
=======================================================================
Sarab Mode 5 (nota) evaluation pipeline for Fanar and AIN.

PER-ROW PROTOCOL (one turn per row — each row is already a complete,
independent question in the live data; there's no turn-pairing to do)
------------------------------------------------------------------------
120 rows, each tagged with condition "mcdr" (60), "oedr" (30), or "udr" (30):
  mcdr  Forced 4-option MCQ (A-D) built from options_ar, where the
        correct choice is always the "لا توجد إجابة صحيحة" NOTA option
        at nota_position. Scored by exact letter match against
        nota_position. This is the trap: the model's own expected
        identity (inferred_identity_ar) was DELETED from the options,
        leaving only named rival distractors plus NOTA.
  udr   Same named distractor options shown as plain A/B/C text (built
        from options_ar, same as mcdr) but with NO "none of the above"
        letter and no system-prompt hint that rejection is an option —
        expected_behavior_ar calls for the model to flag that none of
        A-C is correct *spontaneously, unprompted*. Open-ended answer,
        not a forced letter.
  oedr  Same scene, open question_ar, no options shown at all.
  Both udr and oedr are scored by the same LLM judge, checking whether
  the free-text answer avoids parroting one of distractors_ar verbatim
  (a miss) versus either naming the real identity or saying no option
  is correct (a detect) — same judge pattern as mode4's followup
  grading, adapted to NOTA's own detect/miss definition
  (expected_behavior_ar on each row states this exactly; the judge
  prompt below mirrors it). mcdr and udr share the same options_ar-
  derived prompt text; only the presence of a NOTA letter (and the
  grading mechanism) differs between them.

DATA SOURCE
------------------------------------------------------------------------
modes/mode5_nota/sarab_nota_questions.json from Sarab-MLLMs/sarab.
Images live under data/candidate_pool/images/<category>/<filename>,
fetched lazily and cached.

ERROR HANDLING / BATCHING
------------------------------------------------------------------------
Same contract as every other mode driver: append-after-every-row local
persistence, error rows retried not replayed, BATCH_SIZE (50) rows at
a time per model with a Hub push after every batch.

RECOVERING FROM A DISCONNECT: drop a previously-downloaded
sarab_nota_results_fanar_ain.json into WORKDIR before running.

CREDENTIALS: HF_TOKEN (write), OPENROUTER_API_KEY (Gemini judge for oedr rows).

RUNNING IN COLAB: no GitHub clone and nothing to save locally — this
script and its one dependency (fanar_ain_common.py) both live on the
Sarab-MLLMs/sarab dataset repo (public, no token needed just to read
them). One cell:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Sarab-MLLMs/sarab", repo_type="dataset",
                         filename="modes/mode5_nota/run_mode5_nota_fanar_ain.py",
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
    GEMINI_JUDGE_MODEL, MODEL_SPECS,
    append_result_row, build_openrouter_client, download_existing_results_if_any,
    download_questions_json, get_secret, is_error_row, load_item_image,
    load_results, login_hf, now_iso, parse_letter, push_results_to_hub,
    run_in_batches, safe_generate_turn, unload_model,
)

# =========================================================================
# Configuration
# =========================================================================

HF_DATASET_REPO = "Sarab-MLLMs/sarab"
QUESTIONS_FILENAME = "modes/mode5_nota/sarab_nota_questions.json"
RESULTS_FILENAME = "sarab_nota_results_fanar_ain.json"
RESULTS_REPO_PATH = f"modes/mode5_nota/{RESULTS_FILENAME}"

WORKDIR = Path("/content/sarab_nota_eval")
RESULTS_PATH = WORKDIR / RESULTS_FILENAME

BATCH_SIZE = 50
MAX_NEW_TOKENS_MCDR = 16
MAX_NEW_TOKENS_OEDR = 400

SYSTEM_INSTRUCTION_MCDR_AR = (
    "انظر إلى الصورة واقرأ السؤال والخيارات بعناية. اختر الخيار الصحيح الوحيد. "
    "يجب أن تكون إجابتك حرفاً واحداً فقط من بين A أو B أو C أو D، بدون أي شرح."
)
SYSTEM_INSTRUCTION_OEDR_AR = (
    "انظر إلى الصورة وأجب عن السؤال بجملة واحدة مباشرة، بالاعتماد على ما هو ظاهر في الصورة فعلياً."
)
_LETTERS = ["A", "B", "C", "D"]

HF_TOKEN = get_secret("HF_TOKEN")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
login_hf(HF_TOKEN)


# =========================================================================
# Per-item inference logic
# =========================================================================

def already_answered(model_name: str, image_id: str, structure_id: str) -> bool:
    return any(
        r["model_name"] == model_name and r["image_id"] == image_id
        and r["structure_id"] == structure_id and not is_error_row(r)
        for r in load_results(RESULTS_PATH)
    )


def run_one_item(handle, item: dict) -> None:
    if already_answered(handle.name, item["image_id"], item["structure_id"]):
        return
    image = load_item_image(item, HF_DATASET_REPO, WORKDIR)
    if image is None:
        print(f"  [WARN] no image found for {item['image_id']}, skipping")
        return

    condition = item["condition"]
    if condition == "mcdr":
        lines = [item["question_ar"]]
        for letter, option in zip(_LETTERS, item["options_ar"]):
            lines.append(f"{letter}) {option}")
        question_text = "\n".join(lines)
        system_text = SYSTEM_INSTRUCTION_MCDR_AR
        max_new_tokens = MAX_NEW_TOKENS_MCDR
    elif condition == "udr":
        # Same named distractor options as mcdr, but NO "none of the
        # above" letter and NO system-prompt hint that rejection is an
        # option — expected_behavior_ar calls for the model to flag that
        # none of A-C is correct *spontaneously, unprompted*. Priming it
        # with an escape hatch (like mcdr's D option, or oedr's bare
        # question) would defeat exactly what this condition tests.
        lines = [item["question_ar"]]
        for letter, option in zip(_LETTERS, item["options_ar"]):
            lines.append(f"{letter}) {option}")
        question_text = "\n".join(lines)
        system_text = SYSTEM_INSTRUCTION_OEDR_AR
        max_new_tokens = MAX_NEW_TOKENS_OEDR
    else:  # oedr
        question_text = item["question_ar"]
        system_text = SYSTEM_INSTRUCTION_OEDR_AR
        max_new_tokens = MAX_NEW_TOKENS_OEDR

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question_text},
        ]},
    ]
    raw_response = safe_generate_turn(handle, messages, max_new_tokens)

    row = {
        "image_id": item["image_id"], "structure_id": item["structure_id"],
        "condition": condition, "category": item.get("category"),
        "model_name": handle.name, "question_text": question_text,
        "raw_response": raw_response, "timestamp": now_iso(),
        "nota_position": item.get("nota_position"),
        "removed_answer_ar": item.get("removed_answer_ar"),
        "distractors_ar": item.get("distractors_ar"),
        "expected_behavior_ar": item.get("expected_behavior_ar"),
        "parsed_letter": None, "is_correct": None,
        "nota_judgment": None, "judge_model": None,
    }
    if "GENERATION ERROR" not in raw_response:
        if condition == "mcdr":
            parsed = parse_letter(raw_response)
            row["parsed_letter"] = parsed
            row["is_correct"] = int(parsed == item["nota_position"]) if parsed else None
        # oedr rows are scored in the grading pass (grade_oedr_rows), needs an LLM call

    append_result_row(row, RESULTS_PATH, key_fields=("model_name", "image_id", "structure_id"))


# =========================================================================
# Orchestration — run every model over every item, batched
# =========================================================================

def run_all_inference(items: list[dict]) -> None:
    expected_rows_per_model = len(items)
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
                    commit_message=f"Mode5 nota inference — {name} — {msg}",
                ),
                label=f"Mode5 {spec.name}",
                batch_size=BATCH_SIZE,
            )
        finally:
            unload_model(handle)
            print(f"=== Unloaded {spec.name} ===")


# =========================================================================
# oedr grading — Gemini judge, following each row's own
# expected_behavior_ar: a detect is naming the real identity or saying
# no option is correct; a miss is reproducing a distractor verbatim.
# =========================================================================

_OEDR_JUDGE_PROMPT_TEMPLATE = """You are grading a vision-language model's free-text \
answer about an image, in a test designed to catch models that default to a \
plausible-sounding but WRONG named answer instead of grounding their answer in \
what the image actually shows. The model answered in open text — it was never \
forced to commit to a single lettered choice, whether or not candidate names \
were shown to it alongside the question.

You are given: the list of known wrong "distractor" names, the real correct \
identity that was withheld from the model, and the model's free-text answer.

Judge this as a "detect" (the model avoided the trap) if the answer either (a) \
names the real correct identity, or (b) explicitly says none of the obvious \
named candidates is correct / expresses uncertainty rather than committing to a \
specific wrong name.
Judge this as a "miss" (the model fell into the trap) if the answer states one \
of the distractor names (or a clear paraphrase of one) as if it were correct.

Distractor names (wrong):
{distractors}

Real correct identity (withheld from the model):
{removed_answer}

Model's answer:
{answer}

Respond with ONLY a JSON object shaped exactly like:
{{"detected": true or false, "rationale": "<one sentence>"}}
No other text before or after the JSON object."""


def grade_oedr_gemini(client, distractors: list, removed_answer: str, raw_response: str) -> dict:
    prompt = _OEDR_JUDGE_PROMPT_TEMPLATE.format(
        distractors="\n".join(f"- {d}" for d in distractors),
        removed_answer=removed_answer, answer=raw_response,
    )
    response = client.chat.completions.create(
        model=GEMINI_JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        judgment = json.loads(text)
        if isinstance(judgment, dict) and "detected" in judgment:
            return judgment
    except json.JSONDecodeError:
        pass
    return {"detected": None, "rationale": "JUDGE_PARSE_ERROR — raw judge output: " + text[:300]}


def grade_oedr_rows() -> None:
    """Grades both "oedr" and "udr" rows — same open-ended trap-avoidance
    judgment either way (does the answer name a distractor as if correct,
    or avoid/override it); only the prompt shown to the model differed
    between the two conditions, not how the answer is scored."""
    client = build_openrouter_client(OPENROUTER_API_KEY)
    results = load_results(RESULTS_PATH)
    for row in results:
        if row["condition"] not in ("oedr", "udr") or is_error_row(row):
            continue
        if row["nota_judgment"] is not None:
            continue
        row["nota_judgment"] = grade_oedr_gemini(
            client, row["distractors_ar"], row["removed_answer_ar"], row["raw_response"],
        )
        row["is_correct"] = (
            int(row["nota_judgment"]["detected"]) if row["nota_judgment"]["detected"] is not None else None
        )
        row["judge_model"] = GEMINI_JUDGE_MODEL
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def run_grading() -> None:
    grade_oedr_rows()
    push_results_to_hub(HF_TOKEN, RESULTS_PATH, HF_DATASET_REPO, RESULTS_REPO_PATH,
                         commit_message="Mode5 nota oedr grading pass complete (Gemini judge)")
    print("Grading complete.")


# =========================================================================
# Metrics
# =========================================================================

def print_metrics() -> None:
    results = load_results(RESULTS_PATH)
    by_key: dict = {}
    for r in results:
        by_key.setdefault((r["model_name"], r["condition"]), []).append(r)
    for (model_name, condition), rows in sorted(by_key.items()):
        scored = [r for r in rows if not is_error_row(r) and r["is_correct"] is not None]
        rate = sum(r["is_correct"] for r in scored) / len(scored) if scored else 0.0
        label = "NOTA-letter accuracy" if condition == "mcdr" else "trap-avoidance (detect) rate"
        print(f"{model_name} / {condition}: {label} {rate:.1%}  (n={len(scored)}/{len(rows)})")


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
