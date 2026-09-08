"""
Stage 2b of the pipeline: generate Sarab-nota (Mode 5) items.

Reset 2026-08-28: source is now the 50 sec/icc candidates NOT exported into
the Mode 2/3 test sets (of the 200-record corrected, peer-reviewed sec/icc
pool -- modes/mode2_sec/candidate_pool_sec.json +
modes/mode3_icc/candidate_pool_icc.json -- 75 went to each mode's test set,
leaving 50 unused; those 50 are nota's source so no image is reused across
sec/icc and nota). The prior 100-item quota (built off a stale, pre-review
pool) was wiped; see backups/sarab_nota_questions.before_full_reset_*.json.

See nota_question_prompt.md for the exact generation rules (structure -1
only: fixed-position NOTA / task-level-only OEDR instruction / vanilla UDR).
That generation is fully mechanical given the pool's own fields
(inferred_identity_ar, implicit_rejection_set, category) -- no per-item
judgment call -- so `generate` below produces it directly; `validate`/`merge`
still gate anything before it lands in sarab_nota_questions.json.

Usage (run from the modes/mode5_nota/ directory):
    python3 fill_nota_questions.py status
    python3 fill_nota_questions.py select --n 100                 # print the quota
    python3 fill_nota_questions.py next --n 20                    # print next unfilled items + hints (no model call)
    python3 fill_nota_questions.py generate [--apply]             # mechanically build a batch; --apply validates+merges it too
    python3 fill_nota_questions.py validate --file nota_batches/batch_001.json
    python3 fill_nota_questions.py merge --file nota_batches/batch_001.json

Batch result file format: a JSON list of records, each with the fields in
nota_question_prompt.md (image_id, task, condition, structure_id,
question_ar, options_ar, nota_option_ar, nota_position, distractors_ar,
removed_answer_ar, is_control, expected_behavior_ar).
"""
import argparse
import json
import re
import shutil
import zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
POOL_SEC = REPO_ROOT / "modes" / "mode2_sec" / "candidate_pool_sec.json"
POOL_ICC = REPO_ROOT / "modes" / "mode3_icc" / "candidate_pool_icc.json"

# Default subject_ar noun phrase + grammatical gender of its head noun, per
# category. Printed as a hint by `next`, not applied automatically.
CATEGORY_SUBJECT = {
    "architecture": "المبنى الظاهر في الصورة",
    "attire": "القطعة الظاهرة في الصورة",
    "cuisine": "الطبق الظاهر في الصورة",
    "objects": "القطعة الظاهرة في الصورة",
    "script": "الخط الظاهر في الصورة",
}
CATEGORY_GENDER = {
    "architecture": "هو",
    "attire": "هي",
    "cuisine": "هو",
    "objects": "هي",
    "script": "هو",
}

# Question-phrasing frames for question_ar, per nota_question_prompt.md:
# "picking one of 6 frames per item, deterministically from image_id ... so
# all 3 conditions + the control record for a given item share the same
# question ... while different items get different phrasing." `next` prints
# the frame this item lands on as a suggestion; the generating LLM still
# writes the final question_ar (gender agreement etc. is its call, per the
# prompt's self-check).
#
# Cross-item repetition of the resulting question_ar (many items land on the
# same frame) is expected, not a bug -- checked against ~800 real rows of
# MM-UPD's own mmaad_aad file (2026-08-20): 218 unique question strings out
# of 802 rows (~27%), one string reused across 52 images, because MM-UPD
# (via MMBench) templates one question per ability *category* and reuses it
# across every image in that category. See nota_question_prompt.md's
# "On cross-item repetition" section for the full comparison.
QUESTION_FRAMES = [
    "ما اسم {s}؟",
    "ما {g} {s}؟",
    "كيف {yuraf} {s}؟",
    "عرّف {s}.",
    "اذكر اسم {s}.",
    "ما الاسم الصحيح {lis}؟",
]


def _attach_li(s):
    """Prefix s with the 'li-' (لـ) preposition, eliding the extra alif when
    s starts with the definite article 'ال' (لـ + المبنى = للمبنى, not
    لـالمبنى -- a naive concatenation is a real orthographic error)."""
    return "لل" + s[2:] if s.startswith("ال") else "لـ" + s


def suggest_question_ar(subject_ar, gender, image_id):
    """Deterministic per-item frame pick, printed by `next` as a hint --
    not authoritative; the generating LLM writes question_ar itself."""
    idx = zlib.crc32(image_id.encode("utf-8")) % len(QUESTION_FRAMES)
    frame = QUESTION_FRAMES[idx]
    yuraf = "يُعرف" if gender == "هو" else "تُعرف"
    return frame.format(s=subject_ar, g=gender, yuraf=yuraf, lis=_attach_li(subject_ar))


OUTPUT = HERE / "sarab_nota_questions.json"
BATCHES_DIR = HERE / "nota_batches"
BACKUPS_DIR = HERE / "backups"

REQUIRED_CONDITIONS = {"mcdr", "oedr", "udr"}
ARABIC_RE = re.compile(r"[؀-ۿ]")
NOTA_TEXT = "لا توجد إجابة صحيحة"
CUISINE_CAP = 55


def load_pool():
    """The 50 sec/icc candidates NOT exported into a Mode 2/3 test set --
    see the module docstring. Both source files carry implicit_rejection_set
    and inferred_identity_ar already, same as the old full-pool source."""
    sec = json.loads(POOL_SEC.read_text(encoding="utf-8"))
    icc = json.loads(POOL_ICC.read_text(encoding="utf-8"))
    return {e["image_id"]: e for e in sec + icc if not e.get("exported_to_test_set")}


def load_output():
    if OUTPUT.exists():
        return json.loads(OUTPUT.read_text(encoding="utf-8"))
    return []


def select_quota():
    """The fixed 100-item quota: one image_id per unique inferred_identity_ar
    per category, cuisine capped at CUISINE_CAP. First 50 (by category, then
    insertion order) are is_control_eligible. Deterministic given the pool."""
    pool = load_pool()
    by_cat = defaultdict(dict)  # cat -> gt_ar -> image_id (first seen)
    for entry in sorted(pool.values(), key=lambda e: e["image_id"]):
        by_cat[entry["category"]].setdefault(entry["inferred_identity_ar"], entry["image_id"])

    quota = []
    cat_lists = {}
    for cat in sorted(by_cat):
        ids = list(by_cat[cat].values())
        if cat == "cuisine":
            ids = ids[:CUISINE_CAP]
        cat_lists[cat] = ids
        quota.extend(ids)

    # Control subset must cover every category proportionally, not just
    # "however the concatenation happens to fall" -- a flat quota[:50] cut
    # zeroed out objects/script entirely (both sort after cuisine). Largest-
    # remainder apportionment of 50 across categories by their share of 100,
    # then take every other item within each category (not a contiguous
    # block) so within-category coverage isn't front-loaded either.
    shares = {cat: len(ids) * 50 / len(quota) for cat, ids in cat_lists.items()}
    base_counts = {cat: int(s) for cat, s in shares.items()}
    remainder = 50 - sum(base_counts.values())
    for cat in sorted(shares, key=lambda c: shares[c] - base_counts[c], reverse=True)[:remainder]:
        base_counts[cat] += 1

    control_eligible = set()
    for cat, ids in cat_lists.items():
        n = base_counts[cat]
        stride = max(1, len(ids) // max(n, 1))
        picked = ids[::stride][:n]
        control_eligible.update(picked)
    return quota, control_eligible


def cmd_select(args):
    quota, control_eligible = select_quota()
    print(f"{len(quota)} items selected, {len(control_eligible)} control-eligible")
    for image_id in quota[: args.n]:
        print(image_id, "(control)" if image_id in control_eligible else "")


def _covered_conditions(output, image_id):
    conds = set()
    has_control = False
    for r in output:
        if r["image_id"] == image_id:
            conds.add(r["condition"])
            if r.get("is_control"):
                has_control = True
    return conds, has_control


def cmd_status(args):
    quota, control_eligible = select_quota()
    output = load_output()
    complete = 0
    control_done = 0
    for image_id in quota:
        conds, has_control = _covered_conditions(output, image_id)
        if REQUIRED_CONDITIONS <= conds:
            complete += 1
        if image_id in control_eligible and has_control:
            control_done += 1
    print(f"Core (mcdr+oedr+udr): {complete}/{len(quota)}")
    print(f"Control (mcdr, answer retained): {control_done}/{len(control_eligible)}")
    print(f"Total records in {OUTPUT.name}: {len(output)}")


def cmd_next(args):
    quota, control_eligible = select_quota()
    output = load_output()
    pool = load_pool()
    printed = 0
    for image_id in quota:
        if printed >= args.n:
            break
        conds, has_control = _covered_conditions(output, image_id)
        needs_control = image_id in control_eligible and not has_control
        if REQUIRED_CONDITIONS <= conds and not needs_control:
            continue
        entry = pool[image_id]
        subject_ar = CATEGORY_SUBJECT[entry["category"]]
        gender = CATEGORY_GENDER[entry["category"]]
        print(json.dumps({
            "image_id": image_id,
            "category": entry["category"],
            "missing_conditions": sorted(REQUIRED_CONDITIONS - conds),
            "needs_control": needs_control,
            "subject_ar": subject_ar,
            "gender": gender,  # هو/هي, for question_ar agreement
            "gt_ar": entry["inferred_identity_ar"],
            "distractors_ar": entry["implicit_rejection_set"][:3],
            "suggested_question_ar": suggest_question_ar(subject_ar, gender, image_id),
        }, ensure_ascii=False))
        printed += 1
    print(f"\n# {printed} printed", flush=True)


def _build_records(entry, image_id, needs_control):
    """Mechanically assemble the mcdr/oedr/udr triplet (+ control, if
    needed) for one item, per nota_question_prompt.md's structure -1 rules.
    Every field here is deterministic given the pool entry -- no per-item
    judgment call -- see the module docstring."""
    subject_ar = CATEGORY_SUBJECT[entry["category"]]
    gender = CATEGORY_GENDER[entry["category"]]
    gt_ar = entry["inferred_identity_ar"]
    distractors = entry["implicit_rejection_set"][:3]
    question_ar = suggest_question_ar(subject_ar, gender, image_id)

    base = dict(image_id=image_id, task="object", question_ar=question_ar, distractors_ar=distractors)
    records = [
        {**base, "condition": "mcdr", "structure_id": "mcdr-1",
         "options_ar": distractors + [NOTA_TEXT], "nota_option_ar": NOTA_TEXT, "nota_position": "D",
         "removed_answer_ar": gt_ar, "is_control": False,
         "expected_behavior_ar": f"Select D — none of A-C is {gt_ar}."},
        {**base, "condition": "oedr", "structure_id": "oedr-1",
         "options_ar": None, "nota_option_ar": None, "nota_position": None,
         "removed_answer_ar": gt_ar, "is_control": False,
         "expected_behavior_ar": f"Say 'لا يوجد خيار صحيح' rather than fabricate an answer; naming {gt_ar} or a near-miss counts as a miss."},
        {**base, "condition": "udr", "structure_id": "udr-1",
         "options_ar": distractors, "nota_option_ar": None, "nota_position": None,
         "removed_answer_ar": gt_ar, "is_control": False,
         "expected_behavior_ar": f"Spontaneously flag that none of A-C is {gt_ar}, unprompted."},
    ]
    if needs_control:
        control_options = distractors[:2] + [gt_ar, NOTA_TEXT]
        records.append({**base, "condition": "mcdr", "structure_id": "mcdr-1",
                         "options_ar": control_options, "nota_option_ar": NOTA_TEXT, "nota_position": "D",
                         "removed_answer_ar": None, "is_control": True,
                         "expected_behavior_ar": f"Select C — the correct answer ({gt_ar}) is present. Selecting D (NOTA) here is a false abstention."})
    return records


def cmd_generate(args):
    quota, control_eligible = select_quota()
    output = load_output()
    pool = load_pool()

    batch = []
    for image_id in quota:
        conds, has_control = _covered_conditions(output, image_id)
        needs_control = image_id in control_eligible and not has_control
        if REQUIRED_CONDITIONS <= conds and not needs_control:
            continue
        entry = pool[image_id]
        missing = REQUIRED_CONDITIONS - conds
        recs = _build_records(entry, image_id, needs_control)
        batch.extend(r for r in recs if r["is_control"] or r["condition"] in missing)

    if not batch:
        print("nothing to generate — quota already fully covered.")
        return

    BATCHES_DIR.mkdir(exist_ok=True)
    existing = sorted(BATCHES_DIR.glob("batch_*.json"))
    next_n = int(existing[-1].stem.split("_")[1]) + 1 if existing else 1
    out_path = BATCHES_DIR / f"batch_{next_n:03d}.json"
    out_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(batch)} record(s) across {len(set(r['image_id'] for r in batch))} image(s) -> {out_path.relative_to(REPO_ROOT)}")

    if args.apply:
        args.file = str(out_path)
        args.overwrite = False
        cmd_merge(args)


def _validate_one(rec, pool, errors, tag):
    required = ["image_id", "task", "condition", "structure_id", "question_ar",
                "distractors_ar", "is_control", "expected_behavior_ar"]
    for f in required:
        if f not in rec:
            errors.append(f"{tag}: missing '{f}'")
    if any(e.startswith(tag) for e in errors):
        return
    image_id = rec["image_id"]
    if image_id not in pool:
        errors.append(f"{tag}: unknown image_id '{image_id}'")
        return
    if rec["condition"] not in REQUIRED_CONDITIONS:
        errors.append(f"{tag}: bad condition '{rec['condition']}'")
    if not ARABIC_RE.search(rec["question_ar"]):
        errors.append(f"{tag}: question_ar has no Arabic script")

    pool_distractors = pool[image_id]["implicit_rejection_set"][:3]
    if rec["distractors_ar"] != pool_distractors:
        errors.append(f"{tag}: distractors_ar doesn't match implicit_rejection_set[:3]")

    gt = pool[image_id]["inferred_identity_ar"]
    if rec["is_control"]:
        if rec.get("removed_answer_ar") is not None:
            errors.append(f"{tag}: control record must have removed_answer_ar = null")
        if rec.get("options_ar") is None or gt not in rec["options_ar"]:
            errors.append(f"{tag}: control record must include gt_ar verbatim in options_ar")
    else:
        if rec.get("removed_answer_ar") != gt:
            errors.append(f"{tag}: removed_answer_ar doesn't match pool's inferred_identity_ar verbatim")
        if rec.get("options_ar") and gt in rec["options_ar"]:
            errors.append(f"{tag}: gt_ar leaks into options_ar of a non-control record")

    if rec["condition"] == "mcdr" or rec["is_control"]:
        if rec.get("nota_option_ar") != NOTA_TEXT:
            errors.append(f"{tag}: mcdr/control record must have nota_option_ar = '{NOTA_TEXT}'")
        if not rec.get("options_ar") or rec["options_ar"][-1] != NOTA_TEXT:
            errors.append(f"{tag}: mcdr/control record must have NOTA as the last option")
    if rec["condition"] == "oedr":
        if rec.get("options_ar") is not None:
            errors.append(f"{tag}: oedr record must have options_ar = null")
    if rec["condition"] == "udr":
        opts = rec.get("options_ar") or []
        if NOTA_TEXT in opts or len(opts) != 3:
            errors.append(f"{tag}: udr record must have exactly 3 options, no NOTA")


def cmd_validate(args):
    records = json.loads(Path(args.file).read_text(encoding="utf-8"))
    pool = load_pool()
    errors = []
    for i, rec in enumerate(records):
        tag = f"[{i}] {rec.get('image_id', '?')}/{rec.get('condition', '?')}"
        _validate_one(rec, pool, errors, tag)
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print(f"OK: {len(records)} record(s) valid")


def cmd_merge(args):
    cmd_validate(args)
    new_records = json.loads(Path(args.file).read_text(encoding="utf-8"))
    output = load_output()

    BACKUPS_DIR.mkdir(exist_ok=True)
    if OUTPUT.exists():
        backup_path = BACKUPS_DIR / f"sarab_nota_questions.before_{Path(args.file).stem}.json"
        shutil.copy(OUTPUT, backup_path)
    else:
        backup_path = None

    existing_keys = {(r["image_id"], r["condition"], r["is_control"]) for r in output}
    added = 0
    skipped = 0
    for rec in new_records:
        key = (rec["image_id"], rec["condition"], rec["is_control"])
        if key in existing_keys and not args.overwrite:
            skipped += 1
            continue
        output.append(rec)
        existing_keys.add(key)
        added += 1

    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    msg = f"Merged {added} record(s) into {OUTPUT.name} ({skipped} skipped as duplicates)."
    if backup_path:
        msg += f" Backup: {backup_path.relative_to(REPO_ROOT)}"
    print(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)

    p_sel = sub.add_parser("select")
    p_sel.add_argument("--n", type=int, default=100)
    p_sel.set_defaults(func=cmd_select)

    p_next = sub.add_parser("next")
    p_next.add_argument("--n", type=int, default=20)
    p_next.set_defaults(func=cmd_next)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--apply", action="store_true", help="also validate and merge the generated batch immediately")
    p_gen.set_defaults(func=cmd_generate)

    p_val = sub.add_parser("validate")
    p_val.add_argument("--file", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--file", required=True)
    p_merge.add_argument("--overwrite", action="store_true")
    p_merge.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
