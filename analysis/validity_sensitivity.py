"""
Item-validity sensitivity analysis. Read-only: saved result CSVs and the hand review in analysis/allfail_review.csv.

    python3 analysis/validity_sensitivity.py     # writes analysis/results/validity_sensitivity.md

During the image-by-image review, three kinds of item problem showed up: (1) the true label and the 'no' distractor overlap
(for example warak enab vs mahshi), (2) the question asks about something a photo cannot show (wheat vs sorghum flour,
a person's nationality), (3) the distinction is provenance only (Algerian vs Egyptian garment). This script asks whether
those items change the results by recomputing the main numbers with them removed.
Ground rules are applied to ALL 270 Ground images by pattern; Sway and False exclusions use only the reviewed items.
"""
import csv, importlib.util, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis as ma

HERE = Path(__file__).resolve().parent
OVERLAP = {("ورق عنب", "محشي"), ("ورق عنب", "كوسا محشي"), ("مجبوس", "كبسة"), ("كبسة", "مجبوس"), ("مكدوس", "محشي"),
           ("عصيدة", "بازين"), ("فتة حمص", "حمص"), ("عصيدة بالدجاج", "عصيدة"), ("نقاب", "غطاء الرأس"),
           ("صفيحة لحم", "لحم بعجين"), ("عباية", "الثوب الفلسطيني المطرز")}
PERSON = re.compile(r"هل (?:هذا|هذه) (?:الرجل|رجل|المرأة|امرأة|السيدة|الشخص)")
AMBIGUOUS_UDR_ITEMS = (12, 15, 18, 19, 20, 22, 23)              # positions in the sorted image-id order of the Blank set


def acc(rows):
    ok = sum(ma.is_ok(r["ground_truth"], ma.lenient_parse(r)[1]) for r in rows)
    return ok / len(rows) if rows else float("nan")


def main():
    data = ma.load(); md = []; models = ma.MODEL_ORDER
    review = list(csv.DictReader(open(HERE / "allfail_review.csv", encoding="utf-8")))
    bad = defaultdict(set)                                       # mode -> image_ids with a doubtful label/question or unjudged
    for r in review:
        if r["verdict"].startswith(("DOUBT", "UNSURE")):
            bad[r["mode"]].add(r["image_id"])

    # ---------- Ground ----------
    ref = data[(models[0], 1)]
    overlap_imgs = {r["image_id"] for r in ref if (r["gt_ar"], r["h_item_ar"]) in OVERLAP}
    person_imgs = {r["image_id"] for r in ref if PERSON.search(r["question_text"])}
    attr_imgs = {r["image_id"] for r in ref if r.get("hitem_level") == "attribute"}
    scenarios = [("all 270 images", set()),
                 ("without overlapping-label images (%d)" % len(overlap_imgs), overlap_imgs),
                 ("without those and person-nationality images (%d more)" % len(person_imgs - overlap_imgs), overlap_imgs | person_imgs),
                 ("without those and all attribute-level images (%d more)" % len((attr_imgs | person_imgs | overlap_imgs) - overlap_imgs - person_imgs), overlap_imgs | person_imgs | attr_imgs),
                 ("without every reviewed doubtful image (%d)" % len(bad["Ground"]), bad["Ground"])]
    md.append("## 1. Ground accuracy as problem items are removed\n")
    md.append("| Scenario (images kept) | " + " | ".join(ma.SHORT[m] for m in models) + " | Ranking |"); md.append("|" + "---|" * (len(models) + 2))
    base_rank = None
    for name, excl in scenarios:
        accs = [acc([r for r in data[(m, 1)] if r["image_id"] not in excl]) for m in models]
        kept = len({r["image_id"] for r in ref} - excl)
        rank = " > ".join(ma.SHORT[models[i]].split()[0] for i in np.argsort(accs)[::-1])
        base_rank = base_rank or rank
        md.append("| %s: %d | %s | %s |" % (name, kept, " | ".join("%.1f%%" % (100 * a) for a in accs), rank))
    md.append("\nRanking is by accuracy, best first (names shortened to the first word).\n")

    # ---------- Sway / False ----------
    for mode, label in ((2, "Sway"), (3, "False")):
        excl = bad[label]
        md.append("\n## %d. %s with the reviewed doubtful images removed (%d of 75 images)\n" % (mode, label, len(excl)))
        md.append("| " + " | ".join(ma.SHORT[m] for m in models) + " |"); md.append("|" + "---|" * len(models))
        md.append("| all 75 images: " + " | ".join("%.1f%%" % (100 * acc(data[(m, mode)])) for m in models) + " |")
        md.append("| %d images kept: " % (75 - len(excl)) + " | ".join("%.1f%%" % (100 * acc([r for r in data[(m, mode)] if r["image_id"] not in excl])) for m in models) + " |")
        md.append("\nOnly images that every model fails were reviewed, so removing the doubtful ones raises every model; the table shows whether the ordering holds.\n")

    # ---------- Blank ----------
    spec = importlib.util.spec_from_file_location("m5", ma.ROOT / "modes/mode5_nota/run_mode5_openrouter.py")
    m5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m5)
    ids = sorted({r["image_id"] for r in m5.load_records()})
    drop = {ids[i - 1] for i in AMBIGUOUS_UDR_ITEMS}
    md.append("\n## 4. Blank with the 7 ambiguous items removed (%d images kept of 30)\n" % (len(ids) - len(drop)))
    md.append("| Model | MCDR all -> kept | OEDR all -> kept | UDR all -> kept | False abstention all -> kept |"); md.append("|---|---|---|---|---|")
    for m in models:
        rows = data[(m, 5)]
        def rate(cond, control, excl, flip=False):
            rs = [r for r in rows if r["condition"] == cond and (r["is_control"] == "1") == control and r["image_id"] not in excl]
            v = sum(int(r["is_correct"]) for r in rs) / len(rs)
            return 1 - v if flip else v
        cells = []
        for cond, control, flip in (("mcdr", False, False), ("oedr", False, False), ("udr", False, False), ("mcdr", True, True)):
            cells.append("%.0f%% -> %.0f%%" % (100 * rate(cond, control, set(), flip), 100 * rate(cond, control, drop, flip)))
        md.append("| %s | %s |" % (ma.SHORT[m], " | ".join(cells)))
    text = "\n".join(md)
    (ma.OUT / "validity_sensitivity.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
