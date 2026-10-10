"""
Run the Sarab benchmark for many models x modes at once, and summarise it.

Wraps the five per-mode scripts (mode1_base ... mode5_nota), which stay the
single source of truth for prompts, scoring and metrics. This file only
schedules them, retries failed tasks once, logs each job and prints one table.

Usage (from anywhere):
    export OPENROUTER_API_KEY=sk-or-...        # or put it in .env at the repo root

    python3 modes/run_all.py run --dry-run                    # show the plan, no API calls
    python3 modes/run_all.py run --limit 5                    # smoke test: 5 tasks per mode per model
    python3 modes/run_all.py run                              # full benchmark, all default models
    python3 modes/run_all.py run --models google/gemini-2.5-flash --modes 1 2
    python3 modes/run_all.py run --batch 20                   # 20 calls at a time, each batch checked before the next
    python3 modes/run_all.py run --batch 20 --lanes 3         # same, 3 batches at once (faster)
    python3 modes/run_all.py summary                          # table from the saved results, no API calls

Runs are resumable: finished tasks are skipped, failed ones are retried, so
re-running the same command after an interruption is always safe.
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DEFAULT_MODELS = [
    "google/gemini-2.5-flash",
    "openai/gpt-5-mini",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta/muse-glimmer-30b",
    "mistralai/mistral-large-4-0",
    "z-ai/glm-5.3-flash",
]

# mode number -> (folder, script, results folder, results-file glob, extra run args, expected tasks, label)
MODES = {
    1: ("mode1_base", "run_mode1_openrouter.py", "mode1_results", "results_mode1_full_*.csv", ["--run-label", "full"], 540, "Ground"),
    2: ("mode2_sec", "run_mode2_openrouter.py", "mode2_results", "results_mode2_sec_*.csv", [], 150, "Sway"),
    3: ("mode3_icc", "run_mode3_openrouter.py", "mode3_results", "results_mode3_icc_*.csv", [], 150, "False"),
    4: ("mode4_ccs", "run_mode4_openrouter.py", "mode4_results", "results_mode4_ccs_*.csv", [], 30, "Clash"),
    5: ("mode5_nota", "run_mode5_openrouter.py", "mode5_results", "results_mode5_*.csv", [], 120, "Blank"),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def slug(model):
    """Same file-name slug the mode scripts use for results files."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model)


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def key_info(key):
    """(usage_usd, remaining_usd or None) for the key, or None if unreachable."""
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                     headers={"Authorization": f"Bearer {key}"})
        d = json.load(urllib.request.urlopen(req, timeout=20))["data"]
        return d.get("usage", 0.0), d.get("limit_remaining")
    except Exception:
        return None


def results_file(mode, model):
    folder, _, rdir, pattern, _, _, _ = MODES[mode]
    d = HERE / folder / rdir
    # results files are named results_modeN[_label]_<slug>.csv
    for p in d.glob(pattern):
        if p.stem.endswith(slug(model)):
            return p
    return None


def latest_rows(path):
    """Latest row per task_id (result files are append-only)."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["task_id"]] = r
    return rows


def failed_count(mode, model):
    p = results_file(mode, model)
    if not p:
        return None
    return sum(1 for r in latest_rows(p).values() if r.get("error_type"))


def load_mode_module(mode):
    folder, script = MODES[mode][0], MODES[mode][1]
    spec = importlib.util.spec_from_file_location(f"sarab_mode{mode}", HERE / folder / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run_job(mode, model, args, log_dir, extra=None):
    folder, script, _, _, run_args, _, label = MODES[mode]
    cmd = [sys.executable, "-W", "ignore", script, "run", *run_args, "--model", model,
           "--concurrency", str(args.concurrency)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd.append("--dry-run")
    log = log_dir / f"mode{mode}_{slug(model)}{extra or ''}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=HERE / folder, stdout=fh, stderr=subprocess.STDOUT).returncode
    return mode, model, rc, time.time() - t0, log


# ---------------------------------------------------------------------------
# batched run: N calls at a time, every batch is checked before the next starts
# ---------------------------------------------------------------------------
OPTIONAL_BLANK = ("error_type", "detail")   # blank on success by design
NORMAL_FINISH = ("stop", "end_turn", "eos", "completed")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_number(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def check_rows(mode, rows):
    """Validate the rows one batch wrote.
    hard = unusable row (API error, empty reply, a blank cell);
    soft = usable but odd (cut off, no explanation, answer not parsed, no usage reported)."""
    hard, soft, spend = [], [], 0.0
    for r in rows:
        tid = r.get("task_id", "?")[-36:]
        h, w = [], []
        if r.get("error_type"):
            h.append(r["error_type"])
        reply = (r.get("raw_response") or "").strip()
        if not reply:
            h.append("empty reply")
        blanks = [k for k, v in r.items() if k not in OPTIONAL_BLANK and not str(v).strip()]
        if blanks:
            h.append("blank cells: " + ",".join(blanks[:4]))
        if reply:
            if str(r.get("finish_reason", "")).lower() not in NORMAL_FINISH:
                w.append(f"finish_reason={r.get('finish_reason')}")
            if len(reply.split()) < 6:
                w.append("no explanation")
            if mode < 5:
                if r.get("parsed_answer") not in ("yes", "no"):
                    w.append("answer not parsed (reply does not start with نعم/لا)")
            elif r.get("condition") in ("mcdr",) and not str(r.get("parsed_letter", "")).strip() in ("A", "B", "C", "D"):
                w.append("no option letter parsed")
            if not all(is_number(r.get(k)) for k in ("prompt_tokens", "completion_tokens", "cost_usd")):
                w.append("no usage/cost reported")
        if is_number(r.get("cost_usd")):
            spend += float(r["cost_usd"])
        if h:
            hard.append((tid, "; ".join(h)))
        if w:
            soft.append((tid, "; ".join(w)))
    return hard, soft, spend


def run_batch(mode, model, args, log_dir):
    """Run the next --batch unfinished tasks of one (model, mode) job and validate what was written."""
    folder, script, _, _, run_args, _, label = MODES[mode]
    p = results_file(mode, model)
    prior = read_rows(p) if p else []
    before = len(prior)
    prior_ids = {r["task_id"] for r in prior}
    cmd = [sys.executable, "-W", "ignore", script, "run", *run_args, "--model", model,
           "--concurrency", str(args.batch), "--max-new", str(args.batch)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE / folder, capture_output=True, text=True)
    with open(log_dir / f"mode{mode}_{slug(model)}.log", "a", encoding="utf-8") as fh:
        fh.write(proc.stdout + proc.stderr + "\n--- batch end ---\n")
    m = re.search(r"remaining: (\d+)", proc.stdout)
    remaining_before = int(m.group(1)) if m else 0
    p = results_file(mode, model)
    rows = read_rows(p)[before:] if p else []
    hard, soft, spend = check_rows(mode, rows)
    ok_rows = len(rows) - len(hard)
    fresh = [r for r in rows if r["task_id"] not in prior_ids]       # never attempted before this batch
    fresh_ids = {r["task_id"][-36:] for r in fresh}
    fresh_hard = sum(1 for tid, _ in hard if tid in fresh_ids)
    return {"mode": mode, "model": model, "n": len(rows), "hard": hard, "soft": soft, "spend": spend,
            "fresh_n": len(fresh), "fresh_hard": fresh_hard,
            "ok": ok_rows, "secs": time.time() - t0, "rc": proc.returncode,
            "remaining_after": max(remaining_before - ok_rows, 0) if rows else 0,
            "fatal": any(("api_error_401" in h[1] or "api_error_402" in h[1]) for h in hard)}


def cmd_run_batches(args, models, modes, log_dir, key):
    mods = {mo: load_mode_module(mo) for mo in modes}
    jobs = {}
    for mo in modes:
        for m in models:
            p = results_file(mo, m)
            done = sum(1 for r in latest_rows(p).values() if mods[mo].is_final(r)) if p else 0
            jobs[(m, mo)] = {"remaining": max(MODES[mo][5] - done, 0), "gave_up": False}
    total_tasks = sum(j["remaining"] for j in jobs.values())
    total_batches = sum(-(-j["remaining"] // args.batch) for j in jobs.values())
    print(f"batch mode: {args.batch} calls per batch, {args.lanes} batch(es) at a time, stop if >= "
          f"{args.stop_rate:.0%} of a batch is unusable")
    print(f"remaining: {total_tasks} tasks in about {total_batches} batches across {len(jobs)} jobs")
    if args.dry_run:
        print("DRY RUN - nothing was sent.")
        return
    pending = [k for k, j in jobs.items() if j["remaining"] > 0]
    batch_no, spend_total, stopped, t0 = 0, 0.0, False, time.time()
    bl = open(log_dir / "batches.jsonl", "a", encoding="utf-8")
    while pending and not stopped:
        wave = pending[:args.lanes]
        with ThreadPoolExecutor(max_workers=len(wave)) as ex:
            results = list(ex.map(lambda k: run_batch(k[1], k[0], args, log_dir), wave))
        finished = []
        for (m, mo), res in zip(wave, results):
            batch_no += 1
            spend_total += res["spend"]
            j = jobs[(m, mo)]
            j["remaining"] = res["remaining_after"]
            # the stop rule looks only at tasks never tried before; a batch of retries that fail again
            # (e.g. an image a provider always blocks) just marks the job as given up
            rate = res["fresh_hard"] / res["fresh_n"] if res["fresh_n"] else 0.0
            if res["n"] == 0 or res["remaining_after"] <= 0:
                finished.append((m, mo))
            if res["n"] and res["ok"] == 0 and res["fresh_n"] == 0:
                j["gave_up"] = True
                finished.append((m, mo))
            if res["fatal"] or rate >= args.stop_rate:
                verdict, stopped = "STOP", True
            elif res["hard"]:
                verdict = f"WARN {len(res['hard'])} unusable"
            else:
                verdict = f"OK {res['ok']}/{res['n']} clean, all fields filled"
            soft = f" | {len(res['soft'])} soft warning(s)" if res["soft"] else ""
            print(f"[batch {batch_no}/{total_batches}] mode{mo} {MODES[mo][6]:<6} {m:<30} {res['n']:>2} calls | "
                  f"{verdict}{soft} | ${res['spend']:.4f} | {res['secs']:.0f}s | job left {j['remaining']}", flush=True)
            for tid, why in (res["hard"] + res["soft"])[:3]:
                print(f"      - {tid}: {why}", flush=True)
            bl.write(json.dumps({"batch": batch_no, "model": m, "mode": mo, "n": res["n"], "ok": res["ok"],
                                 "hard": res["hard"], "soft": res["soft"], "spend": res["spend"]},
                                ensure_ascii=False) + "\n")
            bl.flush()
        pending = [k for k in pending[len(wave):] + wave if k not in finished]
        if stopped:
            print("STOPPED: a batch looked wrong (or hit an API/credit error). Fix the cause, then run the same "
                  "command again; finished tasks are skipped.", flush=True)
    bl.close()
    gave_up = [k for k, j in jobs.items() if j["gave_up"]]
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min | {batch_no} batches | logged spend ${spend_total:.4f}")
    if gave_up:
        print("jobs with tasks that kept failing (left for a manual look):", sorted(gave_up))
    after = key_info(key) if key else None
    if after and after[1] is not None:
        print(f"key: ${after[1]:.2f} left")
    print_summary(models, modes)


def cmd_run(args):
    models = args.models or DEFAULT_MODELS
    modes = sorted(args.modes or MODES)
    key = load_api_key()
    if not args.dry_run and not key:
        raise SystemExit("OPENROUTER_API_KEY not found. export it or add it to .env at the repo root.")
    before = key_info(key) if key and not args.dry_run else None
    if before and before[1] is not None:
        print(f"key: ${before[1]:.2f} remaining")
        if before[1] < 2:
            print("WARNING: less than $2 left on this key; a full run can need ~$4-8.")

    jobs = [(m, mo) for mo in modes for m in models]
    log_dir = HERE / "_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(models)} models x {len(modes)} modes = {len(jobs)} jobs "
          f"(parallel {args.parallel}, per-job concurrency {args.concurrency}"
          f"{', limit ' + str(args.limit) if args.limit else ''}{', DRY RUN' if args.dry_run else ''})")
    print(f"logs: {log_dir}")
    if args.batch:
        return cmd_run_batches(args, models, modes, log_dir, key)

    def run_pass(job_list, tag=""):
        failures = []
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(run_job, mo, m, args, log_dir, tag) for m, mo in job_list]
            for f in as_completed(futs):
                mo, m, rc, secs, log = f.result()
                status = "ok" if rc == 0 else f"EXIT {rc}"
                print(f"  mode{mo} {MODES[mo][6]:<6} {m:<32} {status:<8} {secs:6.0f}s")
                if rc != 0:
                    failures.append((m, mo))
        return failures

    t0 = time.time()
    crashed = run_pass(jobs)
    if not args.dry_run and not args.no_retry:
        again = [(m, mo) for m, mo in jobs if (failed_count(mo, m) or 0) > 0]
        if again:
            print(f"\nretrying {len(again)} job(s) that still have failed tasks ...")
            crashed += run_pass(again, tag="_retry")
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")

    after = key_info(key) if key and not args.dry_run else None
    if before and after:
        print(f"spend this run: ${after[0] - before[0]:.4f}"
              + (f" | ${after[1]:.2f} left" if after[1] is not None else ""))
    if crashed:
        print("jobs that exited with an error (see logs):", sorted(set(crashed)))
    if not args.dry_run:
        print_summary(models, modes)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def pct(x):
    return f"{x * 100:.0f}%"


def summarise(models, modes):
    """rows = {model: {mode: dict}} using each mode script's own compute_metrics."""
    out = {m: {} for m in models}
    for mo in modes:
        mod = load_mode_module(mo)
        for m in models:
            p = results_file(mo, m)
            met = mod.compute_metrics(str(p)) if p else None
            if not met:
                continue
            out[m][mo] = met
    return out


def logged_cost(models, modes):
    """{model: (tasks_with_cost, total_usd, prompt_tokens, completion_tokens)} from the per-call logging."""
    out = {}
    for m in models:
        n = usd = pt = ct = 0
        for mo in modes:
            p = results_file(mo, m)
            if not p:
                continue
            for r in latest_rows(p).values():
                try:
                    usd += float(r["cost_usd"]); pt += int(r["prompt_tokens"]); ct += int(r["completion_tokens"]); n += 1
                except (KeyError, ValueError):
                    pass  # rows from before logging, or calls with no response
        out[m] = (n, usd, pt, ct)
    return out


def print_summary(models, modes, write=True):
    data = summarise(models, modes)
    head = ["model"]
    for mo in modes:
        lab = MODES[mo][6]
        head += [f"{lab} done"] + ({5: ["MCDR", "OEDR", "UDR", "false-abst."]}.get(mo) or
                                   (["acc", "PhD", "CATR"] if mo in (2, 3) else ["acc", "PhD", "yes-bias"]))
    rows = []
    for m in models:
        row = [m]
        for mo in modes:
            met = data[m].get(mo)
            exp = MODES[mo][5]
            ncols = 4 if mo == 5 else 3
            if not met:
                row += [f"0/{exp}"] + ["-"] * ncols
                continue
            if mo == 5:
                n = met["n_total"]
                row += [f"{n}/{exp}", pct(met["mcdr"]), pct(met["oedr"]), pct(met["udr"]), pct(met["false_abstention"])]
            else:
                n = met["n_tasks"]
                third = pct(met["catr"]) if mo in (2, 3) else pct(met["yes_rate"])
                row += [f"{n}/{exp}", pct(met["acc_all"]), f"{met['phd_index']:.3f}", third]
        rows.append(row)
    widths = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    md = [line(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows]
    print("\n" + "\n".join(md))
    errs = {m: sum(sum(met.get("error_types", {}).values()) for met in d.values()) for m, d in data.items()}
    if any(errs.values()):
        print("\ntasks whose latest attempt failed:", {m: n for m, n in errs.items() if n})
    cost = logged_cost(models, modes)
    if any(v[0] for v in cost.values()):
        cmd = ["| model | tasks with logged cost | cost (USD) | prompt tokens | completion tokens |", "|---|---|---|---|---|"]
        for m in models:
            n, usd, pt, ct = cost[m]
            cmd.append(f"| {m} | {n} | ${usd:.4f} | {pt:,} | {ct:,} |")
        total = sum(v[1] for v in cost.values())
        cmd.append(f"| **total** | {sum(v[0] for v in cost.values())} | **${total:.4f}** | | |")
        print("\nlogged spend (only calls made after per-call logging was added):\n" + "\n".join(cmd))
        md += ["", "Logged spend (calls made after per-call logging was added):", ""] + cmd
    if write:
        (HERE / "results_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        with open(HERE / "results_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(head); w.writerows(rows)
        print(f"\nsaved: {HERE / 'results_summary.md'} (+ .csv)")


def cmd_summary(args):
    print_summary(args.models or DEFAULT_MODELS, sorted(args.modes or MODES))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("run", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", help="OpenRouter model ids (default: the 6 benchmark models)")
        p.add_argument("--modes", nargs="+", type=int, choices=sorted(MODES), help="mode numbers 1-5 (default: all)")
    r = sub.choices["run"]
    r.add_argument("--limit", type=int, help="cap tasks per mode per model (smoke test)")
    r.add_argument("--concurrency", type=int, default=6, help="parallel requests inside each job (default 6)")
    r.add_argument("--parallel", type=int, default=10, help="jobs running at once (default 10)")
    r.add_argument("--batch", type=int, default=0, help="run in batches of this many calls (e.g. 20); each batch is checked before the next starts")
    r.add_argument("--lanes", type=int, default=1, help="batch mode: batches running at the same time (default 1; raise to go faster)")
    r.add_argument("--stop-rate", type=float, default=0.25, help="batch mode: stop if this share of a batch is unusable (default 0.25)")
    r.add_argument("--no-retry", action="store_true", help="skip the automatic retry pass for failed tasks")
    r.add_argument("--dry-run", action="store_true", help="build tasks and print counts; no API calls")
    args = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}.get(args.cmd or "", lambda a: ap.print_help())(args)


if __name__ == "__main__":
    main()
