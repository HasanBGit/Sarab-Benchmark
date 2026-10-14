"""
Rebuild data/candidate_pool/images/metadata.jsonl and
modes/mode4_ccs/images/metadata.jsonl (the two HF imagefolder metadata files
that drive the Dataset Viewer's "candidate_pool" and "mode4_ccs" configs),
and, as a prerequisite step, patch in any manifest_unified.csv rows that have
real sourcing metadata but were never added to image_pool.json.

These are two SEPARATE configs on purpose, each with only its own real
columns (no cross-schema null padding). An earlier version of this repo
merged both into one "default" config so browsing didn't need a subset
dropdown -- but candidate_pool and mode4_ccs have genuinely different
schemas, so every row ended up null in whichever set of fields belonged to
the other source, which looked broken in the Data Studio viewer. Two clean
configs, zero nulls, one extra click to switch subsets: better trade.

(If you ever want a single merged config again: concatenating two
per-directory metadata.jsonl files with only-partially-overlapping schemas
into one imagefolder split makes the `datasets` loader infer each file's
all-null columns as pyarrow "null" type independently, then fail to cast
that column to "string" when the other file's real values arrive
(TypeError: Couldn't cast array of type string to null). Fixing that needs
one combined root-level metadata.jsonl, not two -- and even then, every row
still has a pile of nulls for the other source's fields, which is the actual
complaint this revert addresses.)

Usage:
    python3 rebuild_metadata_jsonl.py

Idempotent: safe to re-run. The image_pool.json patch step skips any manifest
row whose computed image_id already exists in image_pool.json.
"""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data" / "candidate_pool"
IMAGES_DIR = DATA_DIR / "images"
POOL_FILE = DATA_DIR / "image_pool.json"
MODE1_POOL_FILE = DATA_DIR / "image_pool_mode1.json"
MANIFEST_FILE = DATA_DIR / "manifest_unified.csv"
BACKUP_DIR = REPO_ROOT / "pipeline" / "backups"

CCS_DIR = REPO_ROOT / "modes" / "mode4_ccs" / "images"
CCS_QUESTIONS_FILE = REPO_ROOT / "modes" / "mode4_ccs" / "sarab_ccs_questions.json"
CCS_METADATA_FILE = CCS_DIR / "metadata.jsonl"
POOL_METADATA_FILE = IMAGES_DIR / "metadata.jsonl"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# candidate_pool's own columns (besides file_name/image_id/category, shared by both)
POOL_ONLY_COLUMNS = [
    "subject", "ground_truth_subject", "license", "source",
    "inferred_identity_ar", "inferred_identity_en", "caption_ar", "h_item_ar",
]
# mode4_ccs's own columns
CCS_ONLY_COLUMNS = [
    "image_description_en", "ground_truth_ar", "ground_truth_en", "vqa_trap_ar",
    "vqa_trap_en", "question_yes_ar", "answer_yes", "question_no_ar", "answer_no",
    "failure_mechanism",
]

# Prefer the sibling Sarab-Benchmark checkout's own build_image_pool.py (the
# actual source of truth for slugify()/build_entry()) so this script can
# never drift from it. Vendor an identical fallback only if that repo isn't
# on disk next to this one.
sibling_pipeline = REPO_ROOT.parent / "Sarab-Benchmark" / "pipeline"
if (sibling_pipeline / "build_image_pool.py").exists():
    sys.path.insert(0, str(sibling_pipeline))
    from build_image_pool import slugify, build_entry
else:
    import re

    # Vendored from Sarab-Benchmark/pipeline/build_image_pool.py:40-59
    # (source of truth) -- keep in sync manually if that function changes.
    def slugify(text: str) -> str:
        text = re.sub(r"\.[A-Za-z]+$", "", text)
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return text.lower()

    def none_if_blank(value):
        value = (value or "").strip()
        return value if value else None

    def build_entry(row: dict) -> dict:
        category = row["category"]
        image_id = f"{category}_{slugify(row['filename'])}"
        return {
            "image_id": image_id,
            "filename": row["filename"],
            "local_path": f"data/candidate_pool/images/{category}/{row['filename']}",
            "category": category,
            "subject": none_if_blank(row["subject"]),
            "ground_truth_subject": none_if_blank(row["ground_truth_subject"]),
            "country": none_if_blank(row["country"]),
            "region": none_if_blank(row["region"]),
            "medium": none_if_blank(row["medium"]),
            "license": none_if_blank(row["license"]),
            "artist": none_if_blank(row["artist"]),
            "source": none_if_blank(row["source"]),
            "inferred_identity_ar": None,
            "inferred_identity_en": None,
            "caption_ar": None,
            "h_item_ar": None,
        }


def patch_image_pool():
    """Add any manifest_unified.csv rows with real sourcing metadata that
    aren't in image_pool.json yet (e.g. logged but never captioned/reviewed).
    Skips rows whose file isn't actually present under images/, and rows
    whose computed image_id is already in image_pool.json (idempotent)."""
    pool = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    existing_ids = {r["image_id"] for r in pool}

    with MANIFEST_FILE.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    added = []
    for row in manifest_rows:
        category = row["category"]
        image_id = f"{category}_{slugify(row['filename'])}"
        if image_id in existing_ids:
            continue
        image_path = IMAGES_DIR / category / row["filename"]
        if not image_path.exists():
            continue  # stale manifest row, no matching file on disk
        entry = build_entry(row)
        entry["review_status"] = None  # never captioned/reviewed
        pool.append(entry)
        existing_ids.add(image_id)
        added.append(image_id)

    if added:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(POOL_FILE, BACKUP_DIR / "image_pool.before_add7objects.json")
        POOL_FILE.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Patched image_pool.json: +{len(added)} records -> {len(pool)} total")
        for image_id in added:
            print("  +", image_id)
    else:
        print(f"image_pool.json unchanged ({len(pool)} records, nothing new to add)")

    return pool


def pool_rows(pool):
    """Build candidate_pool's rows (file_name relative to repo root),
    covering every image file under data/candidate_pool/images/. Merges in
    image_pool_mode1.json too, since mode1's images physically live in this
    same directory tree (cataloged separately from image_pool.json)."""
    by_relpath = {}
    for r in pool:
        rel = r["local_path"].split("data/candidate_pool/images/", 1)[1]
        by_relpath[rel] = r
    if MODE1_POOL_FILE.exists():
        mode1_pool = json.loads(MODE1_POOL_FILE.read_text(encoding="utf-8"))
        for r in mode1_pool:
            if not r.get("local_path"):
                continue  # excluded/unfinished candidate, no file to associate
            rel = r["local_path"].split("data/candidate_pool/images/", 1)[1]
            by_relpath.setdefault(rel, r)  # image_pool.json takes priority on overlap

    all_files = []
    for dirpath, _dirs, files in sorted(os.walk(IMAGES_DIR)):
        for fname in sorted(files):
            if Path(fname).suffix.lower() not in IMG_EXT:
                continue
            full = Path(dirpath) / fname
            rel = str(full.relative_to(IMAGES_DIR))
            all_files.append(rel)

    fields = [
        "image_id", "category", "subject", "ground_truth_subject", "license",
        "source", "inferred_identity_ar", "inferred_identity_en", "caption_ar",
        "h_item_ar",
    ]

    rows = []
    n_with_caption = 0
    for rel in all_files:
        category = rel.split("/")[0]
        rec = by_relpath.get(rel)
        # Use the record's own image_id when one exists (it may differ from
        # the slugify default, e.g. a mode's export using a hand-authored
        # id) -- only fall back to computing one for files with no record.
        image_id = rec["image_id"] if rec else f"{category}_{slugify(Path(rel).name)}"
        row = {"file_name": rel, "image_id": image_id}
        if rec:
            for k in fields:
                if k == "image_id":
                    continue
                row[k] = rec.get(k)
            if rec.get("caption_ar"):
                n_with_caption += 1
        else:
            row["category"] = category
            for k in fields:
                if k not in row:
                    row[k] = None
        rows.append(row)

    print(f"candidate_pool: {len(rows)} rows (image_id filled {len(rows)}/{len(rows)}, "
          f"caption_ar filled {n_with_caption}/{len(rows)})")
    return rows


def ccs_rows():
    """Build mode4_ccs's rows (file_name relative to repo root)."""
    questions = json.loads(CCS_QUESTIONS_FILE.read_text(encoding="utf-8"))
    by_filename = {r["filename"]: r for r in questions}

    files = sorted(
        f.name for f in CCS_DIR.iterdir()
        if f.suffix.lower() in IMG_EXT
    )

    fields = ["image_id", "category"] + CCS_ONLY_COLUMNS

    rows = []
    for fname in files:
        row = {"file_name": fname}
        rec = by_filename.get(fname)
        if rec:
            for k in fields:
                row[k] = rec.get(k)
        rows.append(row)

    print(f"mode4_ccs: {len(rows)} rows")
    return rows


def write_metadata(rows, path):
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    pool = patch_image_pool()
    write_metadata(pool_rows(pool), POOL_METADATA_FILE)
    write_metadata(ccs_rows(), CCS_METADATA_FILE)
    root_metadata = REPO_ROOT / "metadata.jsonl"
    if root_metadata.exists():
        root_metadata.unlink()
        print(f"Removed superseded {root_metadata}")


if __name__ == "__main__":
    main()
