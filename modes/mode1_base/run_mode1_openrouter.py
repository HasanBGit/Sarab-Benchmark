"""
Mode 1 (Sarab-base) evaluation via OpenRouter.

Reads mode1 tasks straight from sarab_full_candidate_pool.json (the
single source of truth -- same convention as fill_nota_questions.py, no
separate merged base-questions file) and evaluates them against a
vision-capable model served through OpenRouter's OpenAI-compatible API.

Protocol is frozen prompt_version=1, carried over verbatim from the
original BaseMode.ipynb (Google GenAI SDK) pilot, colocated in this
folder -- same system instruction, same temperature=0, same yes/no-only output contract, same
balanced-sample selection (seed=42) -- so results from this script and
from the notebook are directly comparable. Only the transport changed:
OpenRouter's chat-completions endpoint instead of the Gemini SDK, so any
OpenRouter vision model can be swapped in via --model.

Usage (run from the modes/mode1_base/ directory, or anywhere -- paths are
resolved relative to the repo root):
    export OPENROUTER_API_KEY=sk-or-...      # or put it in a .env file at repo root

    python3 run_mode1_openrouter.py run --run-label sample          # 10 images / 20 tasks
    python3 run_mode1_openrouter.py run --run-label full            # all 270 images / 540 tasks
    python3 run_mode1_openrouter.py run --run-label full --limit 40 # smoke test
    python3 run_mode1_openrouter.py run --dry-run                   # build tasks, no API calls
    python3 run_mode1_openrouter.py metrics --file mode1_results/results_mode1_sample_google-gemini-2.5-flash.csv

    # cross-model comparison table (scans every results CSV in mode1_results/,
    # one row per file, sorted by PhD Index) -- this is what feeds the paper's
    # results table, so read it from here, not recomputed by hand each time:
    python3 run_mode1_openrouter.py table
    python3 run_mode1_openrouter.py table --out mode1_results/comparison_table.md --csv-out mode1_results/comparison_table.csv

Output: one row per task, written incrementally to a CSV under
modes/mode1_base/mode1_results/, resumable across interruptions (only
task_ids that ended in a permanent state -- parsed yes/no, or a
permanent error type -- are skipped on re-run). Reads its tasks from
pipeline/sarab_full_candidate_pool.json (mode=="mode1" records) --
the single source of truth for the merged pool.
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
POOL_FILE = DATA_ROOT / "pipeline" / "sarab_full_candidate_pool.json"
RESULTS_DIR = HERE / "mode1_results"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Frozen protocol (prompt_version = 1) -- do not edit without bumping the
# version and re-running from scratch; changing the prompt invalidates
# comparability with prior runs (Colab pilot and any earlier OpenRouter runs).
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_AR = (
    "انظر إلى الصورة وأجب عن السؤال بالاعتماد على ما هو ظاهر فيها. "
    "يجب أن تكون إجابتك كلمة واحدة فقط: نعم أو لا. "
    "لا تضف أي شرح أو علامات ترقيم أو كلمات أخرى."
)
TEMPERATURE = 0
PROMPT_VERSION = 1
DEFAULT_MODEL = "google/gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 64       # matches the notebook's reasoning-off budget
FALLBACK_MAX_TOKENS = 512    # used if the model won't accept reasoning disabled
SAMPLE_SEED = 42
SAMPLE_TARGET = {"cuisine": 4, "attire": 2, "objects": 2, "architecture": 2}  # 10 images = 20 tasks

FIELDNAMES = [
    "task_id", "image_id", "filename", "category", "polarity", "hitem_level",
    "gt_ar", "h_item_ar", "question_text", "ground_truth", "raw_response",
    "parsed_answer", "is_correct", "error_type", "detail", "attempts",
    "latency_s", "model_name", "provider", "reasoning_disabled",
    "prompt_version", "temperature", "timestamp",
]
PERMANENT_ERRORS = {"image_load_error", "safety_block", "api_error_400", "api_error_403"}

_YES_WORDS = {"نعم", "أجل", "ايوه", "yes"}
_NO_WORDS = {"لا", "كلا", "no"}


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------
def load_mode1_records():
    pool = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    return [r for r in pool if r.get("mode") == "mode1"]


def build_tasks(records):
    tasks = []
    for r in records:
        image_path = str((DATA_ROOT / r["local_path"]).resolve())
        base = dict(
            image_id=r["image_id"], filename=r["filename"], image_path=image_path,
            category=r["category"], hitem_level=r.get("hitem_level"),
            gt_ar=r["inferred_identity_ar"], h_item_ar=r["h_item_ar"],
        )
        tasks.append({**base, "task_id": f"{r['image_id']}__yes", "polarity": "yes",
                      "question_text": r["question_yes_ar"], "ground_truth": r["answer_yes"]})
        tasks.append({**base, "task_id": f"{r['image_id']}__no", "polarity": "no",
                      "question_text": r["question_no_ar"], "ground_truth": r["answer_no"]})
    return tasks


def build_sample_tasks(records, tasks):
    """Balanced 10-image / 20-task sample, seed=42 -- identical selection
    logic to BaseMode.ipynb cell 5, so results are comparable."""
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    for cat, n in SAMPLE_TARGET.items():
        if len(by_cat[cat]) < n:
            raise RuntimeError(f"category {cat!r} has only {len(by_cat[cat])} images, need {n}")

    random.seed(SAMPLE_SEED)
    selected = []
    for cat, n in SAMPLE_TARGET.items():
        pool_cat = by_cat[cat][:]
        random.shuffle(pool_cat)
        selected.extend(pool_cat[:n])

    sel_ids = {r["image_id"] for r in selected}
    if not any(r.get("hitem_level") == "attribute" for r in selected):
        attr_imgs = [r for r in records if r.get("hitem_level") == "attribute" and r["image_id"] not in sel_ids]
        attr_imgs.sort(key=lambda x: (x["category"] != "cuisine", x["image_id"]))
        if attr_imgs:
            repl = attr_imgs[0]
            for i, r in enumerate(selected):
                if r["category"] == "cuisine" and r.get("hitem_level") == "identity":
                    selected[i] = repl
                    break

    sel_ids = {r["image_id"] for r in selected}
    return [t for t in tasks if t["image_id"] in sel_ids]


def preflight(tasks):
    missing = [t for t in tasks if not os.path.isfile(t["image_path"]) or os.path.getsize(t["image_path"]) == 0]
    if missing:
        print(f"❌ preflight failed: {len(missing)} task(s) point at a missing/empty image file:")
        for t in missing[:10]:
            print("   ", t["task_id"], "->", t["image_path"])
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
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
    raise SystemExit(
        "OPENROUTER_API_KEY not found. Either:\n"
        "  export OPENROUTER_API_KEY=sk-or-...\n"
        "or add a line OPENROUTER_API_KEY=sk-or-... to a .env file at the repo root "
        "(already gitignored)."
    )


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def call_model(session, api_key, model, image_path, question_text, max_tokens, reasoning_off, timeout):
    with open(image_path, "rb") as fh:
        img_bytes = fh.read()
    data_uri = f"data:{guess_mime(image_path)};base64,{base64.b64encode(img_bytes).decode('ascii')}"
    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION_AR},
            {"role": "user", "content": [
                {"type": "text", "text": question_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
    }
    if reasoning_off:
        payload["reasoning"] = {"enabled": False}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HasanBGit/Sarab-Benchmark",
        "X-Title": "Sarab Mode 1 (base) evaluation",
    }
    return session.post(API_URL, headers=headers, json=payload, timeout=timeout)


def mk_result(task, model, raw, parsed, error_type, detail, attempts, latency, reasoning_off):
    gt = task["ground_truth"]
    is_correct = int((parsed == "yes" and gt == "نعم") or (parsed == "no" and gt == "لا"))
    return {
        "task_id": task["task_id"], "image_id": task["image_id"], "filename": task["filename"],
        "category": task["category"], "polarity": task["polarity"], "hitem_level": task["hitem_level"],
        "gt_ar": task["gt_ar"], "h_item_ar": task["h_item_ar"], "question_text": task["question_text"],
        "ground_truth": gt, "raw_response": raw, "parsed_answer": parsed, "is_correct": is_correct,
        "error_type": error_type, "detail": detail, "attempts": attempts, "latency_s": latency,
        "model_name": model, "provider": "openrouter", "reasoning_disabled": int(reasoning_off),
        "prompt_version": PROMPT_VERSION, "temperature": TEMPERATURE,
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
            resp = call_model(session, api_key, model, task["image_path"], task["question_text"],
                               max_tokens, reasoning_off, timeout=60)
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            return mk_result(task, model, "", "unclear", "exception", str(e)[:200], attempt,
                              round(time.time() - t0, 2), reasoning_off)

        latency = round(time.time() - t0, 2)

        if resp.status_code == 400:
            try:
                msg = resp.json().get("error", {}).get("message", resp.text)
            except ValueError:
                msg = resp.text
            if "reasoning" in msg.lower() and reasoning_off:
                state[model] = False
                reasoning_off = False
                print(f"   ↪ {model}: rejects reasoning-disabled — retrying with {FALLBACK_MAX_TOKENS} max_tokens")
                continue
            return mk_result(task, model, "", "unclear", "api_error_400", msg[:200], attempt, latency, reasoning_off)

        if resp.status_code == 403:
            return mk_result(task, model, "", "unclear", "api_error_403", resp.text[:200], attempt, latency, reasoning_off)

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue
            et = "rate_limited" if resp.status_code == 429 else f"server_error_{resp.status_code}"
            return mk_result(task, model, "", "unclear", et, resp.text[:200], attempt, latency, reasoning_off)

        if resp.status_code != 200:
            return mk_result(task, model, "", "unclear", f"api_error_{resp.status_code}", resp.text[:200],
                              attempt, latency, reasoning_off)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        raw = ((choice.get("message") or {}).get("content") or "").strip()
        finish_reason = (choice.get("finish_reason") or "").lower()

        if not raw:
            blocked = any(k in finish_reason for k in ("content_filter", "safety"))
            if not blocked and finish_reason == "length" and reasoning_off and attempt < max_retries:
                # reasoning likely ate the whole (small) budget even though "disabled" wasn't honored
                reasoning_off = False
                continue
            return mk_result(task, model, "", "unclear", "safety_block" if blocked else "empty_response",
                              finish_reason, attempt, latency, reasoning_off)

        return mk_result(task, model, raw, parse_answer(raw), "", "", attempt, latency, reasoning_off)

    return mk_result(task, model, "", "unclear", "exhausted_retries", "", max_retries, 0.0, reasoning_off)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------
def is_final(row):
    return row.get("parsed_answer") in ("yes", "no") or row.get("error_type") in PERMANENT_ERRORS


def cmd_run(args):
    records = load_mode1_records()
    if not records:
        raise SystemExit("no mode=='mode1' records found in sarab_full_candidate_pool.json")
    tasks = build_tasks(records)

    if args.run_label == "sample":
        run_tasks = build_sample_tasks(records, tasks)
    else:
        run_tasks = tasks
    if args.limit:
        run_tasks = run_tasks[: args.limit]

    preflight(run_tasks)

    model_slug = re.sub(r"[^a-z0-9.]+", "-", args.model.lower())
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"results_mode1_{args.run_label}_{model_slug}.csv"

    print(f"model={args.model} run_label={args.run_label} tasks={len(run_tasks)} out={out_path}")
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
    todo = [t for t in run_tasks if t["task_id"] not in done]
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
                print(f"[{i}/{len(todo)}] {task['task_id']:<28} {task['polarity']:<3} -> {status}")
                if res["error_type"] == "rate_limited":
                    consec_rate_limited += 1
                    if consec_rate_limited >= 5:
                        print("stopping: 5 consecutive rate-limit errors. Re-run later to resume.")
                        break
                else:
                    consec_rate_limited = 0
        else:
            # Fire `concurrency` requests at the same time via a thread pool.
            # No inter-request pacing (each worker gets its own zeroed
            # last_call, so run_one_task's gap check never sleeps) -- that's
            # the whole point of --concurrency; use --min-gap instead if you
            # want the paced one-at-a-time behavior.
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
                        print(f"[{done_count}/{len(todo)}] {task['task_id']:<28} {task['polarity']:<3} -> {status}")
                        if res["error_type"] == "rate_limited":
                            rate_limited_count += 1
            if rate_limited_count:
                print(f"note: {rate_limited_count}/{len(todo)} tasks hit rate limits -- re-run to resume the rest.")

    print(f"\nsaved results to {out_path}")
    cmd_metrics_from_path(out_path)


# ---------------------------------------------------------------------------
# Metrics (mirrors BaseMode.ipynb cell 8, plus the paper's PhD Index)
# ---------------------------------------------------------------------------
def compute_metrics(path):
    """Read a results CSV and return its metrics as a dict, or None if empty.
    Single source of truth for both `metrics` (one file, full detail) and
    `table` (many files, summary row each) -- so the two commands can never
    disagree about how a number is computed."""
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
    acc_parsed = sum(b(r["is_correct"]) for r in parsed) / len(parsed) if parsed else 0
    yes_recall = sum(b(r["is_correct"]) for r in yes_gt) / len(yes_gt) if yes_gt else 0
    no_recall = sum(b(r["is_correct"]) for r in no_gt) / len(no_gt) if no_gt else 0
    yes_rate = sum(1 for r in parsed if r["parsed_answer"] == "yes") / len(parsed) if parsed else 0
    phd_index = (2 * yes_recall * no_recall / (yes_recall + no_recall)) if (yes_recall + no_recall) > 0 else 0

    by_img = defaultdict(dict)
    for r in rows:
        by_img[r["image_id"]][r["polarity"]] = b(r["is_correct"])
    paired = [d for d in by_img.values() if "yes" in d and "no" in d]
    paired_acc = sum(1 for d in paired if d["yes"] and d["no"]) / len(paired) if paired else 0

    def breakdown(key):
        agg = defaultdict(lambda: [0, 0])
        for r in rows:
            a = agg[r[key]]
            a[0] += b(r["is_correct"])
            a[1] += 1
        return {k: (c / n, n) for k, (c, n) in sorted(agg.items())}

    model_names = {r.get("model_name") for r in rows if r.get("model_name")}
    errs = Counter(r["error_type"] for r in rows if r["error_type"])

    return {
        "path": path, "model_name": next(iter(model_names), "?") if len(model_names) == 1 else "/".join(sorted(model_names)),
        "n_tasks": total, "n_images": len(by_img), "n_parsed": len(parsed), "clarity": len(parsed) / total,
        "acc_all": acc_all, "acc_parsed": acc_parsed, "yes_recall": yes_recall, "no_recall": no_recall,
        "phd_index": phd_index, "yes_rate": yes_rate, "paired_acc": paired_acc, "n_paired": len(paired),
        "by_category": breakdown("category"), "by_hitem_level": breakdown("hitem_level"), "error_types": dict(errs),
    }


def cmd_metrics_from_path(path):
    m = compute_metrics(path)
    if m is None:
        print("no rows.")
        return

    print(f"\n========== {Path(path).name} ==========")
    print(f"tasks: {m['n_tasks']} | parsed: {m['n_parsed']} | unclear/error: {m['n_tasks'] - m['n_parsed']} | clarity: {m['clarity']:.1%}")
    print(f"accuracy (unclear=wrong): {m['acc_all']:.1%}")
    print(f"accuracy (parsed only):   {m['acc_parsed']:.1%}")
    print(f"Yes-recall: {m['yes_recall']:.1%} | No-recall: {m['no_recall']:.1%} | Arabic PhD Index (harmonic mean): {m['phd_index']:.3f}")
    print(f"'yes' bias (50%=neutral): {m['yes_rate']:.1%}")
    print(f"paired (both polarities correct): {m['paired_acc']:.1%} over {m['n_paired']} images")

    print("\nby category:")
    for k, (acc, n) in m["by_category"].items():
        print(f"   {k:<13} {acc:.1%}  (n={n})")
    print("by hitem_level:")
    for k, (acc, n) in m["by_hitem_level"].items():
        print(f"   {k:<10} {acc:.1%}  (n={n})")

    if m["error_types"]:
        print("\nerror types:", m["error_types"])


def cmd_metrics(args):
    cmd_metrics_from_path(Path(args.file))


def cmd_table(args):
    files = sorted(RESULTS_DIR.glob("results_mode1_*.csv"))
    if args.pattern:
        files = [f for f in files if args.pattern in f.name]
    rows = []
    for f in files:
        m = compute_metrics(f)
        if m:
            rows.append(m)
    if not rows:
        print(f"no results CSVs found in {RESULTS_DIR} (pattern={args.pattern!r})")
        return

    rows.sort(key=lambda m: m["phd_index"], reverse=True)

    header = ["model", "n (img/tasks)", "accuracy", "Yes-recall", "No-recall", "PhD Index", "yes-bias", "clarity"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in rows:
        lines.append(
            "| {model} | {ni}/{nt} | {acc:.1%} | {yr:.1%} | {nr:.1%} | {phd:.3f} | {yb:.1%} | {cl:.1%} |".format(
                model=m["model_name"], ni=m["n_images"], nt=m["n_tasks"], acc=m["acc_all"],
                yr=m["yes_recall"], nr=m["no_recall"], phd=m["phd_index"], yb=m["yes_rate"], cl=m["clarity"],
            )
        )
    table_md = "\n".join(lines)
    print(table_md)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(table_md + "\n", encoding="utf-8")
        print(f"\nwritten to {out_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["model_name", "n_images", "n_tasks", "acc_all", "yes_recall",
                                                "no_recall", "phd_index", "yes_rate", "clarity", "paired_acc", "source_file"])
            w.writeheader()
            for m in rows:
                w.writerow({"model_name": m["model_name"], "n_images": m["n_images"], "n_tasks": m["n_tasks"],
                            "acc_all": round(m["acc_all"], 4), "yes_recall": round(m["yes_recall"], 4),
                            "no_recall": round(m["no_recall"], 4), "phd_index": round(m["phd_index"], 4),
                            "yes_rate": round(m["yes_rate"], 4), "clarity": round(m["clarity"], 4),
                            "paired_acc": round(m["paired_acc"], 4), "source_file": m["path"].name})
        print(f"written to {csv_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run Mode 1 evaluation against an OpenRouter model")
    r.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model slug (default: {DEFAULT_MODEL})")
    r.add_argument("--run-label", choices=["sample", "full"], default="sample",
                    help="sample = balanced 10-image/20-task set (seed=42); full = all 270 images/540 tasks")
    r.add_argument("--limit", type=int, default=None, help="cap the number of tasks (after sample/full selection)")
    r.add_argument("--out", default=None, help="output CSV path (default: mode1_results/results_mode1_<label>_<model>.csv)")
    r.add_argument("--min-gap", type=float, default=1.0, help="minimum seconds between requests when --concurrency=1 (default: 1.0)")
    r.add_argument("--concurrency", type=int, default=1, help="run this many requests in parallel via a thread pool (default: 1 = sequential, paced by --min-gap)")
    r.add_argument("--max-retries", type=int, default=5, help="retries per task before giving up (default: 5)")
    r.add_argument("--dry-run", action="store_true", help="build the task list and print counts, no API calls")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("metrics", help="recompute metrics from an existing results CSV")
    m.add_argument("--file", required=True, help="path to a results_mode1_*.csv file")
    m.set_defaults(func=cmd_metrics)

    t = sub.add_parser("table", help="cross-model comparison table across every results CSV in mode1_results/")
    t.add_argument("--pattern", default=None, help="only include files whose name contains this substring")
    t.add_argument("--out", default=None, help="write the markdown table to this path")
    t.add_argument("--csv-out", default=None, help="also write a machine-readable CSV summary to this path")
    t.set_defaults(func=cmd_table)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
