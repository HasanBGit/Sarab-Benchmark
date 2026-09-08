"""
Stage 1b of the pipeline: fill inferred_identity_ar/en, caption_ar, h_item_ar
in image_pool.json. The actual vision judgment is done by Claude (this
session) reading each image directly per figs/candidate_pool/captioning_prompt.md
-- this script is the control layer around that: it selects what's next,
validates what comes back, and merges it in, so the pass is auditable and
resumable instead of one big untracked edit.

Usage (run from the pipeline/ directory):
    python3 fill_captions.py status
    python3 fill_captions.py next --n 10 [--category cuisine]
    python3 fill_captions.py validate --file batches/pilot_001.json
    python3 fill_captions.py merge --file batches/pilot_001.json

Batch result file format: {"image_id": {"inferred_identity_ar": "...",
"inferred_identity_en": "...", "caption_ar": "...", "h_item_ar": "..."}, ...}
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


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
DATA_DIR = DATA_ROOT / "data" / "candidate_pool"
POOL = DATA_DIR / "image_pool.json"
BATCHES_DIR = DATA_ROOT / "pipeline" / "batches"
BACKUPS_DIR = DATA_DIR / "backups"

FIELDS = ["inferred_identity_ar", "inferred_identity_en", "caption_ar", "h_item_ar"]
ARABIC_RE = re.compile(r"[؀-ۿ]")


def load_pool():
    return json.loads(POOL.read_text(encoding="utf-8"))


def is_unfilled(entry):
    return any(entry.get(f) is None for f in FIELDS)


def cmd_status(args):
    pool = load_pool()
    by_cat = {}
    for e in pool:
        cat = e["category"]
        d = by_cat.setdefault(cat, {"total": 0, "filled": 0})
        d["total"] += 1
        if not is_unfilled(e):
            d["filled"] += 1
    total = len(pool)
    filled = sum(1 for e in pool if not is_unfilled(e))
    print(f"Overall: {filled}/{total} filled")
    for cat, d in sorted(by_cat.items()):
        print(f"  {cat:14s} {d['filled']:3d}/{d['total']:3d}")


def cmd_next(args):
    pool = load_pool()
    candidates = [e for e in pool if is_unfilled(e)]
    if args.category:
        candidates = [e for e in candidates if e["category"] == args.category]
    selected = candidates[: args.n]
    for e in selected:
        print(json.dumps({
            "image_id": e["image_id"],
            "local_path": e["local_path"],
            "category": e["category"],
            "hints": {
                "subject": e["subject"],
                "ground_truth_subject": e["ground_truth_subject"],
                "country": e["country"],
                "region": e["region"],
                "medium": e["medium"],
                "source": e["source"],
            },
        }, ensure_ascii=False))
    print(f"\n# {len(selected)} selected ({len(candidates)} unfilled remain in scope)",
          flush=True)


def _validate_one(image_id, rec, errors):
    for f in FIELDS:
        if f not in rec or not isinstance(rec[f], str) or not rec[f].strip():
            errors.append(f"{image_id}: missing/empty '{f}'")
    if len(errors) and errors[-1].startswith(image_id):
        return  # already broken, skip deeper checks
    for f in ["inferred_identity_ar", "caption_ar", "h_item_ar"]:
        if not ARABIC_RE.search(rec[f]):
            errors.append(f"{image_id}: '{f}' contains no Arabic script")
    if not (50 <= len(rec["caption_ar"]) <= 1200):
        errors.append(f"{image_id}: caption_ar length {len(rec['caption_ar'])} outside [50,1200]")
    if rec["inferred_identity_ar"].strip() in rec["caption_ar"]:
        errors.append(f"{image_id}: caption_ar leaks inferred_identity_ar verbatim")
    if rec["h_item_ar"].strip() == rec["inferred_identity_ar"].strip():
        errors.append(f"{image_id}: h_item_ar equals inferred_identity_ar (must be a different foil)")


def cmd_validate(args):
    results = json.loads(Path(args.file).read_text(encoding="utf-8"))
    pool_ids = {e["image_id"] for e in load_pool()}
    errors = []
    unknown = [iid for iid in results if iid not in pool_ids]
    if unknown:
        errors.append(f"unknown image_id(s) not in image_pool.json: {unknown}")
    for image_id, rec in results.items():
        _validate_one(image_id, rec, errors)
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print(f"OK: {len(results)} record(s) valid")


def cmd_merge(args):
    cmd_validate(args)  # re-validate immediately before writing, never trust a stale check
    results = json.loads(Path(args.file).read_text(encoding="utf-8"))
    pool = load_pool()

    BACKUPS_DIR.mkdir(exist_ok=True)
    backup_path = BACKUPS_DIR / f"image_pool.before_{Path(args.file).stem}.json"
    shutil.copy(POOL, backup_path)

    by_id = {e["image_id"]: e for e in pool}
    updated = 0
    overwritten = 0
    for image_id, rec in results.items():
        entry = by_id[image_id]
        if not is_unfilled(entry) and not args.overwrite:
            print(f"skip (already filled, pass --overwrite to replace): {image_id}")
            continue
        if not is_unfilled(entry):
            overwritten += 1
        for f in FIELDS:
            entry[f] = rec[f]
        updated += 1

    POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {updated} record(s) into {POOL.name} ({overwritten} overwritten). "
          f"Backup: {backup_path.relative_to(REPO_ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)

    p_next = sub.add_parser("next")
    p_next.add_argument("--n", type=int, default=10)
    p_next.add_argument("--category", default=None,
                         choices=["architecture", "attire", "cuisine", "objects", "script"])
    p_next.set_defaults(func=cmd_next)

    p_val = sub.add_parser("validate")
    p_val.add_argument("--file", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--file", required=True)
    p_merge.add_argument("--overwrite", action="store_true",
                          help="also replace entries that already have all 4 fields filled")
    p_merge.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
