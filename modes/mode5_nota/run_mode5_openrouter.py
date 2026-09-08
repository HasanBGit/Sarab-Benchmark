"""
Mode 5 (Sarab-nota, Absent Answer Detection) evaluation via OpenRouter.

Evaluates all 120 records in sarab_nota_questions.json (30 images x
mcdr/oedr/udr + control). Three different prompt/response formats per
condition, matching nota_question_prompt.md and Wang et al. 2026's
MCDR/OEDR/UDR protocol:

  mcdr / control -- options_ar shown as a lettered list (A-D, NOTA always
    last). Model told to answer with exactly one Latin letter. Correct
    letter is derived per-record (whichever option isn't a listed
    distractor and isn't the NOTA text), not hardcoded to "D"/"C" -- so a
    reordering of the source data doesn't silently break scoring.

  udr -- 3 real distractor options (A-C) shown as an ordinary-looking MCQ,
    with the SAME rigid "answer one letter only" instruction as mcdr --
    deliberately no escape hatch. Wang et al.'s UDR condition tests
    whether a model spontaneously breaks that instruction to flag nothing
    fits; softening the instruction here would just turn it into OEDR.
    Scoring: a clean A/B/C letter = compliant = miss (didn't detect);
    anything else (hedged/refused/flagged) = detected.

  oedr -- no options shown at all, free-text answer, task-level permission
    stated once ("...you may say 'no correct option'"), per
    nota_question_prompt.md. Scoring: fabricating one of the 3 distractors
    verbatim = miss; anything else (a hedge, or genuinely naming the real
    answer since the model can still see the image) = detected/correct.

Usage (run from modes/mode5_nota/, or anywhere -- paths resolve from repo root):
    export OPENROUTER_API_KEY=sk-or-...
    python3 run_mode5_openrouter.py run --dry-run
    python3 run_mode5_openrouter.py run --concurrency 30
    python3 run_mode5_openrouter.py run --model openai/gpt-4o-mini --concurrency 30
    python3 run_mode5_openrouter.py metrics --file mode5_results/results_mode5_google-gemini-2.5-flash.csv
    python3 run_mode5_openrouter.py table
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
from collections import Counter, defaultdict
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
    return Path(snapshot_download(repo_id="Sarab-MLLMs/sarab", repo_type="dataset"))


DATA_ROOT = _resolve_data_root()
NOTA_FILE = DATA_ROOT / "modes" / "mode5_nota" / "sarab_nota_questions.json"
RESULTS_DIR = HERE / "mode5_results"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

TEMPERATURE = 0
PROMPT_VERSION = 1
DEFAULT_MODEL = "google/gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 64
FALLBACK_MAX_TOKENS = 512
NOTA_TEXT = "لا توجد إجابة صحيحة"
LETTERS = ["A", "B", "C", "D"]

BASE_INSTRUCTION_AR = "انظر إلى الصورة وأجب عن السؤال بالاعتماد على ما هو ظاهر فيها."
MCQ_INSTRUCTION_AR = "أجب بحرف لاتيني واحد فقط يطابق الخيار الصحيح، دون أي شرح أو علامات ترقيم."
# Verbatim from nota_question_prompt.md's OEDR section -- deliberately a
# different phrase from NOTA_TEXT (the MCDR option string): "لا يوجد خيار
# صحيح" here, not "لا توجد إجابة صحيحة". Match the doc exactly, don't
# normalize the two into one phrase.
OEDR_INSTRUCTION_AR = (
    "أجب عن السؤال بإجابة واحدة مختصرة. إن لم تكن متأكدًا من وجود إجابة صحيحة "
    "تتوافق مع الصورة، يمكنك القول: \"لا يوجد خيار صحيح\"."
)

FIELDNAMES = [
    "task_id", "image_id", "condition", "structure_id", "is_control",
    "question_text", "options_shown", "correct_letter", "raw_response",
    "parsed_letter", "is_correct", "error_type", "detail", "attempts",
    "latency_s", "model_name", "provider", "reasoning_disabled",
    "prompt_version", "temperature", "timestamp",
]
PERMANENT_ERRORS = {"image_load_error", "safety_block", "api_error_400", "api_error_403"}


# ---------------------------------------------------------------------------
def load_records():
    return json.loads(NOTA_FILE.read_text(encoding="utf-8"))


def resolve_image_path(image_id):
    """nota records don't carry local_path themselves -- look it up from the
    two mode2/mode3 candidate pools they were sourced from."""
    for f in ("modes/mode2_sec/candidate_pool_sec.json", "modes/mode3_icc/candidate_pool_icc.json"):
        for r in json.loads((DATA_ROOT / f).read_text(encoding="utf-8")):
            if r["image_id"] == image_id:
                return str((DATA_ROOT / r["local_path"]).resolve())
    return None


def build_tasks(records, limit=None):
    image_paths = {}
    tasks = []
    for r in records:
        iid = r["image_id"]
        if iid not in image_paths:
            image_paths[iid] = resolve_image_path(iid)
        task = dict(r)
        task["task_id"] = f"{iid}__{r['condition']}__{'control' if r['is_control'] else 'core'}"
        task["image_path"] = image_paths[iid]
        tasks.append(task)
    if limit:
        tasks = tasks[:limit]
    return tasks


def preflight(tasks):
    missing = [t for t in tasks if not t["image_path"] or not os.path.isfile(t["image_path"])]
    if missing:
        missing_ids = {t["task_id"] for t in missing}
        print(f"⚠️  skipping {len(missing)} task(s) with a missing image file:")
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


def guess_mime(path):
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/jpeg"


# ---------------------------------------------------------------------------
# Prompt assembly + correctness derivation, per condition
# ---------------------------------------------------------------------------
def build_prompt(task):
    cond = task["condition"]
    options = task.get("options_ar")

    question = task.get("question_text") or task.get("question_ar")
    if options:  # mcdr, control, udr
        lettered = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        prompt = f"{question}\n{lettered}"
        system = f"{BASE_INSTRUCTION_AR} {MCQ_INSTRUCTION_AR}"
        options_shown = " | ".join(f"{LETTERS[i]}={opt}" for i, opt in enumerate(options))
    else:  # oedr
        prompt = question
        system = f"{BASE_INSTRUCTION_AR} {OEDR_INSTRUCTION_AR}"
        options_shown = ""
    return system, prompt, options_shown


def correct_letter_for(task):
    """mcdr non-control -> letter of the NOTA option. control -> letter of
    whichever option isn't a distractor and isn't NOTA (the reinstated real
    answer). udr/oedr have no single correct letter (see scoring below)."""
    options = task.get("options_ar")
    if not options:
        return None
    distractors = set(task.get("distractors_ar") or [])
    if task["is_control"]:
        for i, opt in enumerate(options):
            if opt not in distractors and opt != NOTA_TEXT:
                return LETTERS[i]
        return None
    if task["condition"] == "mcdr":
        for i, opt in enumerate(options):
            if opt == NOTA_TEXT:
                return LETTERS[i]
    return None


def parse_letter(raw):
    if not raw:
        return None
    text = raw.strip().strip(" \n\t\"'.,!?؟،؛:-")
    m = re.match(r"^([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    # tolerate "Option A" / "الإجابة A" style wrapping without extra prose
    m = re.search(r"\b([A-D])\b", text)
    if m and len(text) <= 20:
        return m.group(1)
    return None


def score(task, raw):
    """Returns (parsed_letter_or_None, is_correct: int)."""
    cond = task["condition"]
    if task["is_control"] or cond == "mcdr":
        letter = parse_letter(raw)
        correct = correct_letter_for(task)
        return letter, int(letter is not None and letter == correct)

    if cond == "udr":
        letter = parse_letter(raw)
        # compliant with the (unhinted) MCQ instruction -> picked a real
        # option -> failed to detect. Anything else -> spontaneous detection.
        return letter, int(letter is None)

    if cond == "oedr":
        text = (raw or "").strip()
        distractors = task.get("distractors_ar") or []
        fabricated = any(d and d in text for d in distractors)
        return None, int(bool(text) and not fabricated)

    return None, 0


# ---------------------------------------------------------------------------
def call_model(session, api_key, model, image_path, system, prompt_text, max_tokens, reasoning_off, timeout):
    with open(image_path, "rb") as fh:
        img_bytes = fh.read()
    data_uri = f"data:{guess_mime(image_path)};base64,{base64.b64encode(img_bytes).decode('ascii')}"
    payload = {
        "model": model, "temperature": TEMPERATURE, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
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
        "X-Title": "Sarab Mode 5 (nota) evaluation",
    }
    return session.post(API_URL, headers=headers, json=payload, timeout=timeout)


def mk_result(task, model, raw, parsed_letter, is_correct, error_type, detail, attempts, latency, reasoning_off, options_shown):
    return {
        "task_id": task["task_id"], "image_id": task["image_id"], "condition": task["condition"],
        "structure_id": task["structure_id"], "is_control": int(task["is_control"]),
        "question_text": task.get("question_text") or task.get("question_ar"), "options_shown": options_shown,
        "correct_letter": correct_letter_for(task) or "", "raw_response": raw,
        "parsed_letter": parsed_letter or "", "is_correct": is_correct,
        "error_type": error_type, "detail": detail, "attempts": attempts, "latency_s": latency,
        "model_name": model, "provider": "openrouter", "reasoning_disabled": int(reasoning_off),
        "prompt_version": PROMPT_VERSION, "temperature": TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_one_task(session, api_key, model, task, min_gap, max_retries, state, last_call):
    if not task["image_path"] or not os.path.isfile(task["image_path"]):
        return mk_result(task, model, "", None, 0, "image_load_error", "file not found", 0, 0.0, True, "")

    system, prompt_text, options_shown = build_prompt(task)
    reasoning_off = state.get(model, True)
    for attempt in range(1, max_retries + 1):
        gap = min_gap - (time.time() - last_call[0])
        if gap > 0:
            time.sleep(gap)
        max_tokens = MAX_OUTPUT_TOKENS if reasoning_off else FALLBACK_MAX_TOKENS
        t0 = time.time()
        last_call[0] = t0
        try:
            resp = call_model(session, api_key, model, task["image_path"], system, prompt_text, max_tokens, reasoning_off, timeout=60)
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            return mk_result(task, model, "", None, 0, "exception", str(e)[:200], attempt, round(time.time() - t0, 2), reasoning_off, options_shown)

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
            return mk_result(task, model, "", None, 0, "api_error_400", msg[:200], attempt, latency, reasoning_off, options_shown)
        if resp.status_code == 403:
            return mk_result(task, model, "", None, 0, "api_error_403", resp.text[:200], attempt, latency, reasoning_off, options_shown)
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            et = "rate_limited" if resp.status_code == 429 else f"server_error_{resp.status_code}"
            return mk_result(task, model, "", None, 0, et, resp.text[:200], attempt, latency, reasoning_off, options_shown)
        if resp.status_code != 200:
            return mk_result(task, model, "", None, 0, f"api_error_{resp.status_code}", resp.text[:200], attempt, latency, reasoning_off, options_shown)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        raw = ((choice.get("message") or {}).get("content") or "").strip()
        finish_reason = (choice.get("finish_reason") or "").lower()
        if not raw:
            blocked = any(k in finish_reason for k in ("content_filter", "safety"))
            if not blocked and finish_reason == "length" and reasoning_off and attempt < max_retries:
                reasoning_off = False
                continue
            return mk_result(task, model, "", None, 0, "safety_block" if blocked else "empty_response", finish_reason, attempt, latency, reasoning_off, options_shown)

        parsed_letter, is_correct = score(task, raw)
        return mk_result(task, model, raw, parsed_letter, is_correct, "", "", attempt, latency, reasoning_off, options_shown)

    return mk_result(task, model, "", None, 0, "exhausted_retries", "", max_retries, 0.0, reasoning_off, options_shown)


def is_final(row):
    return row.get("error_type") in PERMANENT_ERRORS or row.get("raw_response")


def cmd_run(args):
    records = load_records()
    tasks = build_tasks(records, args.limit)
    preflight(tasks)

    model_slug = re.sub(r"[^a-z0-9.]+", "-", args.model.lower())
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"results_mode5_{model_slug}.csv"

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
                status = f"{res['parsed_letter']}({res['is_correct']})" if not res["error_type"] else f"⚠️{res['error_type']}"
                print(f"[{i}/{len(todo)}] {task['task_id']:<40} -> {status}")
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
                        status = f"{res['parsed_letter']}({res['is_correct']})" if not res["error_type"] else f"⚠️{res['error_type']}"
                        print(f"[{done_count}/{len(todo)}] {task['task_id']:<40} -> {status}")
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
    if not rows:
        return None

    def b(x):
        return str(x) == "1"

    def rate(cond, control):
        subset = [r for r in rows if r["condition"] == cond and (str(r["is_control"]) == "1") == control and not r["error_type"]]
        return (sum(b(r["is_correct"]) for r in subset) / len(subset) if subset else 0), len(subset)

    mcdr_core, n_mcdr = rate("mcdr", False)
    oedr_core, n_oedr = rate("oedr", False)
    udr_core, n_udr = rate("udr", False)
    control_rate, n_control = rate("mcdr", True)
    false_abstention = 1 - control_rate  # control should select the real answer, not NOTA

    model_names = {r.get("model_name") for r in rows if r.get("model_name")}
    errs = Counter(r["error_type"] for r in rows if r["error_type"])
    return {
        "path": path, "model_name": next(iter(model_names), "?") if len(model_names) == 1 else "/".join(sorted(model_names)),
        "n_total": len(rows), "n_errors": sum(errs.values()),
        "mcdr": mcdr_core, "n_mcdr": n_mcdr, "oedr": oedr_core, "n_oedr": n_oedr, "udr": udr_core, "n_udr": n_udr,
        "false_abstention": false_abstention, "n_control": n_control, "error_types": dict(errs),
    }


def cmd_metrics_from_path(path):
    m = compute_metrics(path)
    if m is None:
        print("no rows.")
        return
    print(f"\n========== {Path(path).name} ==========")
    print(f"total rows: {m['n_total']} | errors: {m['n_errors']}")
    print(f"MCDR detection: {m['mcdr']:.1%} (n={m['n_mcdr']})")
    print(f"OEDR detection: {m['oedr']:.1%} (n={m['n_oedr']})")
    print(f"UDR  detection: {m['udr']:.1%} (n={m['n_udr']})")
    print(f"False-abstention rate (control set): {m['false_abstention']:.1%} (n={m['n_control']})")
    if m["error_types"]:
        print("error types:", m["error_types"])


def cmd_metrics(args):
    cmd_metrics_from_path(Path(args.file))


def cmd_table(args):
    files = sorted(RESULTS_DIR.glob("results_mode5_*.csv"))
    rows = [m for m in (compute_metrics(f) for f in files) if m]
    if not rows:
        print(f"no results CSVs found in {RESULTS_DIR}")
        return
    rows.sort(key=lambda m: m["mcdr"], reverse=True)
    header = ["model", "MCDR", "OEDR", "UDR", "false-abstention"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in rows:
        lines.append("| {model} | {mcdr:.1%} | {oedr:.1%} | {udr:.1%} | {fa:.1%} |".format(
            model=m["model_name"], mcdr=m["mcdr"], oedr=m["oedr"], udr=m["udr"], fa=m["false_abstention"]))
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
    r.add_argument("--limit", type=int, default=None)
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
