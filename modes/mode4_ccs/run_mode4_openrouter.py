"""
Mode 4 (Sarab-ccs, Cultural Counter-Common-Sense) evaluation via OpenRouter.

Protocol verified directly against Liu et al. 2025 (CVPR), the paper
Sarab-ccs mirrors -- read via info/Liu_PhD_A_ChatGPT-Prompted_Visual_
Hallucination_Evaluation_Dataset_CVPR_2025_paper (1).pdf, secs 3.2-3.5:
  - PhD-ccs is plain binary VQA over CCS (AI-generated, counter-common-sense)
    images, no context text prepended (that's sec/icc, not ccs) -- same
    forced single-word yes/no output and system instruction as Mode 1
    (Sarab-base), just a different image source.
  - Sec. 3.4/Table 3: each CCS description (the norm violation actually
    depicted, e.g. "trees growing underwater") is paired with a CS
    (common-sense) counterpart (e.g. "coral growing underwater") -- CCS
    yields the Yes-question (GT=Yes), CS yields the No-question (GT=No).
    Scored identically to Mode 1: accuracy, Yes-recall, No-recall, PhD
    Index (harmonic mean).

The 15 images delivered for this mode (originally gemini-code-
1787877704755.json, now modes/mode4_ccs/sarab_ccs_questions.json) only
came with the CCS side (an open-ended trap question, vqa_trap_ar/en, plus
a descriptive ground_truth_ar/en of the violation and a failure_mechanism
note) -- no paired CS counterpart, so no Yes/No pair existed yet. An
earlier version of this script scored that data as free-text VQA judged by
a second LLM call (see modes/mode4_ccs/backups/pre_yesno_protocol_20260829/
for that version and its results). To match the paper's actual protocol,
question_yes_ar/answer_yes/question_no_ar/answer_no were added to
sarab_ccs_questions.json for all 15 images: question_yes_ar reformats the
existing CCS claim into strict polar form (GT=نعم), question_no_ar is an
authored CS counterpart -- a specific, plausible "normal" claim that is
NOT depicted (GT=لا), mechanically derived from each record's existing
ground_truth_ar/failure_mechanism (no new facts, just the missing polarity
written out), the same way Mode 1's h_item_ar-based no-questions were
derived. vqa_trap_ar/en, ground_truth_ar/en, and failure_mechanism are kept
on every record as provenance/documentation, not removed.

Usage (run from modes/mode4_ccs/, or anywhere -- paths resolve from repo root):
    export OPENROUTER_API_KEY=sk-or-...      # or a .env file at repo root
    python3 run_mode4_openrouter.py run --dry-run
    python3 run_mode4_openrouter.py run --concurrency 5
    python3 run_mode4_openrouter.py run --model openai/gpt-4o-mini --concurrency 5
    python3 run_mode4_openrouter.py metrics --file mode4_results/results_mode4_ccs_google-gemini-2.5-flash.csv
    python3 run_mode4_openrouter.py table
"""
import argparse
import base64
import csv
import json
import mimetypes
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def _resolve_data_root():
    """Resolve Sarab's data directory: SARAB_DATA_DIR env override first
    (point it at a local checkout while developing), then a sibling
    Sarab-Dataset-HF/ directory next to this repo, then download and cache
    the dataset from the Hugging Face Hub."""
    override = os.environ.get("SARAB_DATA_DIR")
    if override:
        return Path(override)
    sibling = REPO_ROOT.parent / "Sarab-Dataset-HF"
    if sibling.exists():
        return sibling
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id="HassanB4/sarab", repo_type="dataset"))


DATA_ROOT = _resolve_data_root()
QUESTION_SET = DATA_ROOT / "modes" / "mode4_ccs" / "sarab_ccs_questions.json"
RESULTS_DIR = HERE / "mode4_results"
MODE_LABEL = "mode4_ccs"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Frozen protocol (prompt_version = 1) -- same base instruction as Mode 1
# (no context text: PhD-ccs is plain image+question, per the paper).
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_AR = (
    "انظر إلى الصورة وأجب عن السؤال بالاعتماد على ما هو ظاهر فيها. "
    "يجب أن تكون إجابتك كلمة واحدة فقط: نعم أو لا. "
    "لا تضف أي شرح أو علامات ترقيم أو كلمات أخرى."
)
TEMPERATURE = 0
PROMPT_VERSION = 1
DEFAULT_MODEL = "google/gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 64
FALLBACK_MAX_TOKENS = 512

FIELDNAMES = [
    "task_id", "image_id", "filename", "category", "polarity",
    "question_text", "ground_truth", "raw_response",
    "parsed_answer", "is_correct", "error_type", "detail", "attempts",
    "latency_s", "model_name", "provider", "reasoning_disabled",
    "prompt_version", "temperature", "timestamp",
]
PERMANENT_ERRORS = {"image_load_error", "safety_block", "api_error_400"}
# api_error_403 deliberately excluded: OpenRouter returns 403 for both real
# auth failures and "key limit exceeded" (a prepaid-credit cap) -- the
# latter is exactly the kind of thing a re-run after topping up should retry.

_YES_WORDS = {"نعم", "أجل", "ايوه", "yes"}
_NO_WORDS = {"لا", "كلا", "no"}


# ---------------------------------------------------------------------------
def load_question_set():
    return json.loads(QUESTION_SET.read_text(encoding="utf-8"))


def build_tasks(records, limit=None):
    tasks = []
    for r in records:
        image_path = str((DATA_ROOT / r["local_path"]).resolve())
        base = dict(image_id=r["image_id"], filename=r["filename"], image_path=image_path,
                    category=r["category"])
        tasks.append({**base, "task_id": f"{r['image_id']}__yes", "polarity": "yes",
                      "question_text": r["question_yes_ar"], "ground_truth": r["answer_yes"]})
        tasks.append({**base, "task_id": f"{r['image_id']}__no", "polarity": "no",
                      "question_text": r["question_no_ar"], "ground_truth": r["answer_no"]})
    if limit:
        tasks = tasks[:limit]
    return tasks


def preflight(tasks):
    """Skip tasks whose image is missing rather than hard-fail the whole run."""
    missing = [t for t in tasks if not os.path.isfile(t["image_path"]) or os.path.getsize(t["image_path"]) == 0]
    if missing:
        missing_ids = {t["task_id"] for t in missing}
        print(f"⚠️  skipping {len(missing)} task(s) with a missing/empty image file:")
        for t in missing[:20]:
            print("   ", t["task_id"], "->", t["image_path"])
        tasks[:] = [t for t in tasks if t["task_id"] not in missing_ids]
    return tasks


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OPENROUTER_API_KEY":
                return v.strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found. export it or add it to a .env file at the repo root.")


def parse_answer(raw):
    if not raw:
        return "unclear"
    text = raw.strip().strip(" \n\t\"'.,!?؟،؛:-")
    if text in _YES_WORDS or text.lower() in _YES_WORDS:
        return "yes"
    if text in _NO_WORDS or text.lower() in _NO_WORDS:
        return "no"
    tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    has_yes = any(t in _YES_WORDS or t.lower() in _YES_WORDS for t in tokens)
    has_no = any(t in _NO_WORDS or t.lower() in _NO_WORDS for t in tokens)
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unclear"


def guess_mime(path):
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/jpeg"


def call_model(session, api_key, model, image_path, prompt_text, max_tokens, reasoning_off, timeout):
    with open(image_path, "rb") as fh:
        img_bytes = fh.read()
    data_uri = f"data:{guess_mime(image_path)};base64,{base64.b64encode(img_bytes).decode('ascii')}"
    payload = {
        "model": model, "temperature": TEMPERATURE, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION_AR},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
    }
    if reasoning_off:
        payload["reasoning"] = {"enabled": False}
    headers = {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HasanBGit/Sarab-Benchmark",
        "X-Title": f"Sarab {MODE_LABEL} evaluation",
    }
    return session.post(API_URL, headers=headers, json=payload, timeout=timeout)


def mk_result(task, model, raw, parsed, error_type, detail, attempts, latency, reasoning_off):
    gt = task["ground_truth"]
    is_correct = int((parsed == "yes" and gt == "نعم") or (parsed == "no" and gt == "لا"))
    return {
        "task_id": task["task_id"], "image_id": task["image_id"], "filename": task["filename"],
        "category": task["category"], "polarity": task["polarity"],
        "question_text": task["question_text"], "ground_truth": gt, "raw_response": raw,
        "parsed_answer": parsed, "is_correct": is_correct, "error_type": error_type, "detail": detail,
        "attempts": attempts, "latency_s": latency, "model_name": model, "provider": "openrouter",
        "reasoning_disabled": int(reasoning_off), "prompt_version": PROMPT_VERSION, "temperature": TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_one_task(session, api_key, model, task, min_gap, max_retries, state, last_call):
    if not os.path.isfile(task["image_path"]):
        return mk_result(task, model, "", "unclear", "image_load_error", "file not found", 0, 0.0, True)

    reasoning_off = state.get(model, True)
    for attempt in range(1, max_retries + 1):
        gap = min_gap - (time.time() - last_call[0])
        if gap > 0:
            time.sleep(gap)
        max_tokens = MAX_OUTPUT_TOKENS if reasoning_off else FALLBACK_MAX_TOKENS
        t0 = time.time()
        last_call[0] = t0
        try:
            resp = call_model(session, api_key, model, task["image_path"], task["question_text"], max_tokens, reasoning_off, timeout=60)
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            return mk_result(task, model, "", "unclear", "exception", str(e)[:200], attempt, round(time.time() - t0, 2), reasoning_off)

        latency = round(time.time() - t0, 2)

        if resp.status_code == 400:
            try:
                msg = resp.json().get("error", {}).get("message", resp.text)
            except ValueError:
                msg = resp.text
            if "reasoning" in msg.lower() and reasoning_off:
                state[model] = False
                reasoning_off = False
                continue
            return mk_result(task, model, "", "unclear", "api_error_400", msg[:200], attempt, latency, reasoning_off)
        if resp.status_code == 403:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            return mk_result(task, model, "", "unclear", "api_error_403", resp.text[:200], attempt, latency, reasoning_off)
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            et = "rate_limited" if resp.status_code == 429 else f"server_error_{resp.status_code}"
            return mk_result(task, model, "", "unclear", et, resp.text[:200], attempt, latency, reasoning_off)
        if resp.status_code != 200:
            return mk_result(task, model, "", "unclear", f"api_error_{resp.status_code}", resp.text[:200], attempt, latency, reasoning_off)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        raw = ((choice.get("message") or {}).get("content") or "").strip()
        finish_reason = (choice.get("finish_reason") or "").lower()
        if not raw:
            blocked = any(k in finish_reason for k in ("content_filter", "safety"))
            if not blocked and finish_reason == "length" and reasoning_off and attempt < max_retries:
                reasoning_off = False
                continue
            return mk_result(task, model, "", "unclear", "safety_block" if blocked else "empty_response", finish_reason, attempt, latency, reasoning_off)
        return mk_result(task, model, raw, parse_answer(raw), "", "", attempt, latency, reasoning_off)

    return mk_result(task, model, "", "unclear", "exhausted_retries", "", max_retries, 0.0, reasoning_off)


def is_final(row):
    return row.get("parsed_answer") in ("yes", "no") or row.get("error_type") in PERMANENT_ERRORS


def cmd_run(args):
    records = load_question_set()
    tasks = build_tasks(records, args.limit)
    preflight(tasks)

    model_slug = re.sub(r"[^a-z0-9.]+", "-", args.model.lower())
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"results_{MODE_LABEL}_{model_slug}.csv"

    print(f"model={args.model} tasks={len(tasks)} out={out_path}")
    if args.dry_run:
        print("dry run — no API calls made.")
        return

    api_key = load_api_key()
    done = set()
    if out_path.is_file():
        with open(out_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if is_final(row):
                    done.add(row["task_id"])
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"already complete: {len(done)} | remaining: {len(todo)}")
    if not todo:
        print("nothing to do.")
        cmd_metrics_from_path(out_path)
        return

    state = {}
    new_file = not out_path.is_file()
    session = requests.Session()
    write_lock = threading.Lock()

    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()

        if args.concurrency <= 1:
            last_call = [0.0]
            consec_rate_limited = 0
            for i, task in enumerate(todo, 1):
                res = run_one_task(session, api_key, args.model, task, args.min_gap, args.max_retries, state, last_call)
                writer.writerow(res)
                fh.flush()
                status = res["parsed_answer"] if not res["error_type"] else f"⚠️{res['error_type']}"
                print(f"[{i}/{len(todo)}] {task['task_id']:<24} {task['polarity']:<3} -> {status}")
                if res["error_type"] == "rate_limited":
                    consec_rate_limited += 1
                    if consec_rate_limited >= 5:
                        print("stopping: 5 consecutive rate-limit errors. Re-run later to resume.")
                        break
                else:
                    consec_rate_limited = 0
        else:
            print(f"running {len(todo)} tasks with concurrency={args.concurrency} (no pacing between requests)")
            done_count = 0
            rate_limited_count = 0

            def _worker(task):
                return run_one_task(session, api_key, args.model, task, 0.0, args.max_retries, state, [0.0])

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_worker, t): t for t in todo}
                for fut in as_completed(futures):
                    task = futures[fut]
                    res = fut.result()
                    with write_lock:
                        writer.writerow(res)
                        fh.flush()
                        done_count += 1
                        status = res["parsed_answer"] if not res["error_type"] else f"⚠️{res['error_type']}"
                        print(f"[{done_count}/{len(todo)}] {task['task_id']:<24} {task['polarity']:<3} -> {status}")
                        if res["error_type"] == "rate_limited":
                            rate_limited_count += 1
            if rate_limited_count:
                print(f"note: {rate_limited_count}/{len(todo)} tasks hit rate limits -- re-run to resume the rest.")

    print(f"\nsaved results to {out_path}")
    cmd_metrics_from_path(out_path)


# ---------------------------------------------------------------------------
def compute_metrics(path):
    rows_by_task = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows_by_task[row["task_id"]] = row
    rows = list(rows_by_task.values())

    def b(x):
        return str(x) == "1"

    total = len(rows)
    if total == 0:
        return None
    parsed = [r for r in rows if r["parsed_answer"] in ("yes", "no")]
    yes_gt = [r for r in rows if r["ground_truth"] == "نعم"]
    no_gt = [r for r in rows if r["ground_truth"] == "لا"]
    acc_all = sum(b(r["is_correct"]) for r in rows) / total
    yes_recall = sum(b(r["is_correct"]) for r in yes_gt) / len(yes_gt) if yes_gt else 0
    no_recall = sum(b(r["is_correct"]) for r in no_gt) / len(no_gt) if no_gt else 0
    phd_index = (2 * yes_recall * no_recall / (yes_recall + no_recall)) if (yes_recall + no_recall) > 0 else 0
    yes_rate = sum(1 for r in parsed if r["parsed_answer"] == "yes") / len(parsed) if parsed else 0

    model_names = {r.get("model_name") for r in rows if r.get("model_name")}
    return {
        "path": path, "model_name": next(iter(model_names), "?") if len(model_names) == 1 else "/".join(sorted(model_names)),
        "n_tasks": total, "n_parsed": len(parsed), "clarity": len(parsed) / total,
        "acc_all": acc_all, "yes_recall": yes_recall, "no_recall": no_recall,
        "phd_index": phd_index, "yes_rate": yes_rate,
        "error_types": dict(Counter(r["error_type"] for r in rows if r["error_type"])),
    }


def cmd_metrics_from_path(path):
    m = compute_metrics(path)
    if m is None:
        print("no rows.")
        return
    print(f"\n========== {Path(path).name} ==========")
    print(f"tasks: {m['n_tasks']} | parsed: {m['n_parsed']} | clarity: {m['clarity']:.1%}")
    print(f"accuracy: {m['acc_all']:.1%}")
    print(f"Yes-recall: {m['yes_recall']:.1%} | No-recall: {m['no_recall']:.1%} | PhD Index: {m['phd_index']:.3f}")
    print(f"'yes' bias: {m['yes_rate']:.1%}")
    if m["error_types"]:
        print("error types:", m["error_types"])


def cmd_metrics(args):
    cmd_metrics_from_path(Path(args.file))


def cmd_table(args):
    files = sorted(RESULTS_DIR.glob(f"results_{MODE_LABEL}_*.csv"))
    rows = [m for m in (compute_metrics(f) for f in files) if m]
    if not rows:
        print(f"no results CSVs found in {RESULTS_DIR}")
        return
    rows.sort(key=lambda m: m["phd_index"], reverse=True)
    header = ["model", "n", "accuracy", "Yes-recall", "No-recall", "PhD Index", "clarity"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in rows:
        lines.append("| {model} | {n} | {acc:.1%} | {yr:.1%} | {nr:.1%} | {phd:.3f} | {cl:.1%} |".format(
            model=m["model_name"], n=m["n_tasks"], acc=m["acc_all"], yr=m["yes_recall"],
            nr=m["no_recall"], phd=m["phd_index"], cl=m["clarity"]))
    table_md = "\n".join(lines)
    print(table_md)
    if args.out:
        Path(args.out).write_text(table_md + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--limit", type=int, default=None, help="cap number of tasks (2 per image)")
    r.add_argument("--out", default=None)
    r.add_argument("--min-gap", type=float, default=1.0)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--max-retries", type=int, default=5)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("metrics")
    m.add_argument("--file", required=True)
    m.set_defaults(func=cmd_metrics)

    t = sub.add_parser("table")
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_table)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
