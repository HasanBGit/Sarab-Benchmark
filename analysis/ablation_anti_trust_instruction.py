"""
Ablation: does the anti-trust instruction sentence in Sway/False's system
prompt actually change model behavior? The sentence (verbatim from Liu et
al.'s original paper) is: "in case of a conflict between the accompanying
text and the image, the model should follow the image." Every row in the
current release was collected WITH this sentence; this ablation asks what
happens WITHOUT it, everything else held identical (same image, same
misleading/incorrect caption, same question, temperature=0).

Cost design: the "with instruction" arm is NOT re-run. It's read straight
from the already-published, already-paid-for full-run result CSVs for the
same sampled items. Only the "without instruction" arm makes new,
billed OpenRouter calls. This roughly halves the cost of the ablation
versus running both arms fresh.

Budget note: this key had only $0.70 remaining as of 2026-09-08 (checked
via the free /key endpoint, see below). Given that, --n-images and
--models default small; raise them only after checking real per-call cost
with `smoke-test` first.

    curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      https://openrouter.ai/api/v1/key       # free, no cost, shows remaining balance

Usage (run from anywhere -- paths resolve like the mode scripts):
    # 1. One real call, prints exact cost (OpenRouter returns per-request
    #    cost when usage.include=true is set) so you know what a full run
    #    will actually cost before committing to it:
    python3 analysis/ablation_anti_trust_instruction.py smoke-test

    # 2. The real ablation run -- only the "without instruction" arm is new
    #    API traffic; defaults to the 2 cheapest models and 10 images
    #    (20 tasks) per mode = 40 new calls per model = 80 total:
    python3 analysis/ablation_anti_trust_instruction.py run
    python3 analysis/ablation_anti_trust_instruction.py run --n-images 5 --models openai/gpt-4o-mini

    # 3. Build the with-vs-without comparison table from whatever has been
    #    collected so far (safe to re-run any time, makes no API calls):
    python3 analysis/ablation_anti_trust_instruction.py compare
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "modes"))

import mode2_sec.run_mode2_openrouter as mod2  # noqa: E402
import mode3_icc.run_mode3_openrouter as mod3  # noqa: E402

OUT_DIR = HERE / "results" / "ablation_anti_trust"

# The one sentence being ablated (verbatim, including the leading space
# that joins it to the preceding sentence in the frozen instruction).
ANTI_TRUST_SENTENCE = " في حال وجود تعارض بين السياق النصي المرفق ومحتوى الصورة، اعتمد على الصورة."
MODES = {
    "sec": {"mod": mod2, "label": "Sway", "baseline_dir_name": "mode2_results", "baseline_glob": "results_mode2_sec_*.csv"},
    "icc": {"mod": mod3, "label": "False", "baseline_dir_name": "mode3_results", "baseline_glob": "results_mode3_icc_*.csv"},
}

DEFAULT_MODELS = ["google/gemini-2.5-flash-lite", "openai/gpt-4o-mini"]  # cheapest 2 of the paper's 4


def ablated_instruction(mod):
    full = mod.SYSTEM_INSTRUCTION_AR
    assert full.endswith(ANTI_TRUST_SENTENCE), "SYSTEM_INSTRUCTION_AR changed upstream -- update ANTI_TRUST_SENTENCE"
    return full[: -len(ANTI_TRUST_SENTENCE)]


def sample_tasks(mod, n_images, seed=42):
    """Deterministic sample of n_images (2 tasks each: yes + no), restricted
    to images whose file actually exists (mirrors preflight())."""
    records = mod.load_test_set()
    tasks = mod.build_tasks(records)
    mod.preflight(tasks)
    image_ids = sorted({t["image_id"] for t in tasks})
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    chosen = set(image_ids[:n_images])
    return [t for t in tasks if t["image_id"] in chosen]


def run_ablation(models, n_images, min_gap, max_retries):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode_key, info in MODES.items():
        mod = info["mod"]
        tasks = sample_tasks(mod, n_images)
        original_instruction = mod.SYSTEM_INSTRUCTION_AR
        mod.SYSTEM_INSTRUCTION_AR = ablated_instruction(mod)
        try:
            for model in models:
                slug = model.replace("/", "-")
                out_path = OUT_DIR / f"{mode_key}_{slug}_without_instruction.csv"
                done = set()
                if out_path.is_file():
                    with open(out_path, newline="", encoding="utf-8") as fh:
                        for row in csv.DictReader(fh):
                            if mod.is_final(row):
                                done.add(row["task_id"])
                todo = [t for t in tasks if t["task_id"] not in done]
                print(f"[{mode_key}] {model}: {len(done)} already done, {len(todo)} to run")
                if not todo:
                    continue
                api_key = mod.load_api_key()
                session = __import__("requests").Session()
                state = {}
                last_call = [0.0]
                new_file = not out_path.is_file()
                with open(out_path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=mod.FIELDNAMES)
                    if new_file:
                        writer.writeheader()
                    for i, task in enumerate(todo, 1):
                        res = mod.run_one_task(session, api_key, model, task, min_gap, max_retries, state, last_call)
                        writer.writerow(res)
                        fh.flush()
                        status = res["parsed_answer"] if not res["error_type"] else f"err:{res['error_type']}"
                        print(f"  [{i}/{len(todo)}] {task['task_id']:<30} -> {status}")
        finally:
            mod.SYSTEM_INSTRUCTION_AR = original_instruction
    print(f"\nDone. Results under {OUT_DIR}/")


def smoke_test(model, min_gap, max_retries):
    """One real call under the ablated (without-sentence) instruction,
    requesting cost info from OpenRouter so we know exactly what a full
    run will cost before committing to it."""
    import requests

    mod = mod2
    tasks = sample_tasks(mod, n_images=1)
    task = tasks[0]
    original_instruction = mod.SYSTEM_INSTRUCTION_AR
    mod.SYSTEM_INSTRUCTION_AR = ablated_instruction(mod)
    try:
        api_key = mod.load_api_key()
        session = requests.Session()

        # Same request call_model() would build, but with usage.include so
        # OpenRouter reports the real cost of this exact request.
        with open(task["image_path"], "rb") as fh:
            import base64
            img_bytes = fh.read()
        data_uri = f"data:{mod.guess_mime(task['image_path'])};base64,{base64.b64encode(img_bytes).decode('ascii')}"
        prompt_text = f"{task['context_ar']}\n\n{task['question_text']}"
        payload = {
            "model": model, "temperature": 0, "max_tokens": 64,
            "usage": {"include": True},
            "messages": [
                {"role": "system", "content": mod.SYSTEM_INSTRUCTION_AR},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]},
            ],
            "reasoning": {"enabled": False},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        t0 = time.time()
        resp = session.post(mod.API_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - t0
        print(f"task_id={task['task_id']} model={model} status={resp.status_code} latency={latency:.2f}s")
        data = resp.json()
        if resp.status_code == 200:
            choice = (data.get("choices") or [{}])[0]
            raw = ((choice.get("message") or {}).get("content") or "").strip()
            usage = data.get("usage", {})
            cost = usage.get("cost")
            print(f"raw_response={raw!r}")
            print(f"parsed={mod.parse_answer(raw)!r}  ground_truth={task['ground_truth']!r}")
            print(f"usage={usage}")
            if cost is not None:
                print(f"\nActual cost of this one call: ${cost:.6f}")
                print(f"At that rate, N total calls would cost ~${cost} * N")
            else:
                print("\n(no per-request cost field returned -- check usage tokens above against "
                      "this model's per-token price on openrouter.ai/models instead)")
        else:
            print(f"error response: {data}")
    finally:
        mod.SYSTEM_INSTRUCTION_AR = original_instruction


def compare(models):
    """Join each mode's 'without instruction' results (just collected)
    against the matching task_ids in the already-published 'with
    instruction' full-run CSVs, and report accuracy/CATR under both."""
    rows = []
    for mode_key, info in MODES.items():
        baseline_dir = mod2.DATA_ROOT / "modes" / ("mode2_sec" if mode_key == "sec" else "mode3_icc") / info["baseline_dir_name"]
        for model in models:
            slug = model.replace("/", "-")
            without_path = OUT_DIR / f"{mode_key}_{slug}_without_instruction.csv"
            if not without_path.is_file():
                continue
            with open(without_path, newline="", encoding="utf-8") as fh:
                without_rows = {r["task_id"]: r for r in csv.DictReader(fh) if r["parsed_answer"] in ("yes", "no")}
            if not without_rows:
                continue

            baseline_file = None
            for p in sorted(baseline_dir.glob(info["baseline_glob"])):
                with open(p, newline="", encoding="utf-8") as fh:
                    first_model = next(csv.DictReader(fh), {}).get("model_name")
                if first_model == model:
                    baseline_file = p
                    break
            if baseline_file is None:
                print(f"no baseline file found for {model} in {baseline_dir}, skipping {mode_key}")
                continue
            with open(baseline_file, newline="", encoding="utf-8") as fh:
                with_rows = {r["task_id"]: r for r in csv.DictReader(fh) if r["parsed_answer"] in ("yes", "no")}

            paired_ids = sorted(set(without_rows) & set(with_rows))
            if not paired_ids:
                continue

            def acc(rowset, ids):
                b = [str(rowset[i]["is_correct"]) == "1" for i in ids]
                return sum(b) / len(b)

            with_acc = acc(with_rows, paired_ids)
            without_acc = acc(without_rows, paired_ids)
            rows.append({
                "mode": info["label"], "model": model, "n_paired_tasks": len(paired_ids),
                "acc_with_instruction": with_acc, "catr_with_instruction": 1 - with_acc,
                "acc_without_instruction": without_acc, "catr_without_instruction": 1 - without_acc,
                "delta_acc": without_acc - with_acc,
            })

    if not rows:
        print("no paired with/without data yet -- run `run` first.")
        return

    out_csv = OUT_DIR / "comparison.csv"
    out_md = OUT_DIR / "comparison.md"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = ["# Anti-trust instruction ablation: with vs. without", "",
             "| Mode | Model | n | Acc (with) | Acc (without) | CATR (with) | CATR (without) | Δ Acc |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['model']} | {r['n_paired_tasks']} | "
                      f"{r['acc_with_instruction']:.1%} | {r['acc_without_instruction']:.1%} | "
                      f"{r['catr_with_instruction']:.1%} | {r['catr_without_instruction']:.1%} | "
                      f"{r['delta_acc']:+.1%} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwritten to {out_csv} and {out_md}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke-test", help="one real call, prints exact cost before committing to a full run")
    s.add_argument("--model", default=DEFAULT_MODELS[0])
    s.add_argument("--min-gap", type=float, default=1.0)
    s.add_argument("--max-retries", type=int, default=3)

    r = sub.add_parser("run", help="run the 'without instruction' arm (new API calls)")
    r.add_argument("--models", default=",".join(DEFAULT_MODELS))
    r.add_argument("--n-images", type=int, default=10)
    r.add_argument("--min-gap", type=float, default=1.0)
    r.add_argument("--max-retries", type=int, default=3)

    c = sub.add_parser("compare", help="build the with-vs-without comparison table (no API calls)")
    c.add_argument("--models", default=",".join(DEFAULT_MODELS))

    args = p.parse_args()
    if args.cmd == "smoke-test":
        smoke_test(args.model, args.min_gap, args.max_retries)
    elif args.cmd == "run":
        run_ablation(args.models.split(","), args.n_images, args.min_gap, args.max_retries)
    elif args.cmd == "compare":
        compare(args.models.split(","))


if __name__ == "__main__":
    main()
