## Stage 2b: generate Sarab-nota (Mode 5) items from the candidate pool

Source: `pipeline/sarab_full_candidate_pool.json` only — the single source
of truth. Everything this pass needs is already vision-verified there:
`inferred_identity_ar` (→ `gt_ar`), `implicit_rejection_set` (4 grounded
distractors per image, already built), and `category` (→ `subject_ar`/gender
via a fixed per-category lookup table, `CATEGORY_SUBJECT`/`CATEGORY_GENDER`
in `fill_nota_questions.py`). No separate merged base-questions file is read
or produced. Produces the
MCDR/OEDR/UDR absent-answer-detection items described in
`ArabPhD_Modes_and_Recommendation.md` and `ArabPhD_Mode5_Question_Structures.md`.

**Structure used: `-1` only (fixed-position NOTA / task-level-only OEDR
instruction / vanilla UDR) — the literature-matching primary structure per
the verdict in `ArabPhD_Mode5_Pilot_Benchmark.md`, checked against
`wang2026absentanswer` and `miyai2025unsolvable`.** The `-2`/`-3` ablation
structures from the design doc are not generated in this production pass —
they're a separate robustness-check task, not part of getting real Mode-5
coverage.

## Item selection (100 core items)

Not all 281 base images — many are repeat photos of the same landmark/dish/
script style sharing an identical question (e.g. 5 Al Faisaliah Tower
photos). Nota tests genuine detection, not photo memorization, so one item
per **unique identity** (`gt_ar`), not per image:
- architecture: 16/16 unique identities (all)
- attire: 21/21 (all)
- cuisine: 55/86 (capped, so cuisine doesn't dominate the set)
- objects: 2/2 (all)
- script: 6/6 (all)
- **Total: 100**

50 of these 100 are also flagged `is_control_eligible` for the matched
control subset below.

## Per selected item: what gets generated

Three nota records (`condition`: `mcdr`, `oedr`, `udr`), all sharing the same
stripped answer and distractor set, plus one control record for the 50
flagged items.

**Distractors**: first 3 entries of `implicit_rejection_set` (already
grounded per-image, vision-verified during captioning — not regenerated
here).

**Question phrasing** — nota reframes the base yes/no pair as a WH-question
(the MCQ candidate set makes "is it X?" meaningless once 3+ options are
offered). A single fixed frame ("ما هو/هي {subject_ar}؟") was tried first and
produced only 5 distinct question templates across all 100 items — too easy
to key off the surface form. Fixed by picking one of 6 frames per item,
deterministically from `image_id` (`QUESTION_FRAMES` in
`fill_nota_questions.py`), so all 3 conditions + the control record for a
given item share the same question (required — MCDR/OEDR/UDR must ask the
*same* question, only option-visibility differs) while different items get
different phrasing:

> ما اسم {subject}؟ | ما {هو|هي} {subject}؟ | كيف يُعرف/تُعرف {subject}؟ |
> عرّف {subject}. | اذكر اسم {subject}. | ما الاسم الصحيح لـ{subject}؟

(`لـ` + a subject starting with `ال` elides to `لل`, e.g. `للمبنى` not
`لـالمبنى` — a real orthographic error the frame code has to handle, not just
naive string concatenation.)

Gender (`هو`/`هي`, and the `يُعرف`/`تُعرف` verb form) is derived from
`category` via a fixed table (`CATEGORY_GENDER` in `fill_nota_questions.py`)
— never guessed fresh per item.

**On cross-item repetition of `question_ar`** — with 6 frames × 5 categories,
many items land on the identical surface string (e.g. every `attire`/`objects`
item that hits frame 2 asks the exact same `question_ar`). This was checked
against the literature, not assumed: pulling ~800 of the 820 real rows of
MM-UPD's own `mmaad_aad` file (2026-08-20, via HF LFS range request, not just
a small toy sample) found **218
unique question strings out of 802 rows (≈27%)**, with the single most common
string (*"Which one is the correct caption of this image?"*) reused across
**52 different images**. Per the paper's Appendix B.1, this is inherited
structurally: MM-UPD adapts MMBench, which templates one fixed question per
"ability" category (celebrity recognition, object localization, attribute
comparison, …) and reuses it across every image in that bucket — the
image + option set carry the per-item signal, not question-text novelty.
Our 22 unique strings across 100 items (≈22%) is the same order of magnitude
by the same logic (`subject_ar`/frame is a property of the category+item, not
meant to be unique prose per item) — not a defect to fix.

### `mcdr` record

```json
{
  "image_id": "...", "task": "object", "condition": "mcdr", "structure_id": "mcdr-1",
  "question_ar": "ما هو/هي {subject_ar}؟",
  "options_ar": ["{distractor1}", "{distractor2}", "{distractor3}", "لا توجد إجابة صحيحة"],
  "nota_option_ar": "لا توجد إجابة صحيحة", "nota_position": "D",
  "distractors_ar": ["{distractor1}", "{distractor2}", "{distractor3}"],
  "removed_answer_ar": "{gt_ar}", "is_control": false,
  "expected_behavior_ar": "Select D."
}
```

### `oedr` record

Same `question_ar`, `options_ar: null`, no NOTA field shown to the model.
Batch-level instruction (state once, not per item):
> أجب عن كل سؤال بإجابة واحدة مختصرة. إن لم تكن متأكدًا من وجود إجابة صحيحة
> تتوافق مع الصورة، يمكنك القول: "لا يوجد خيار صحيح".

### `udr` record

Same `question_ar`, `options_ar` = the 3 distractors only, no NOTA, no cue.

### control record (50 of the 100 only, `mcdr` form)

Same as the `mcdr` record but `removed_answer_ar` is put back as option A
(distractors shift to B/C/D, NOTA stays a 5th... no — keep 4 options total:
replace one distractor with the real answer so the set stays 4-wide),
`removed_answer_ar: null`, `is_control: true`. Tests false-abstention: a
model selecting NOTA here is wrong.

## Self-check before returning

1. `distractors_ar` are exactly `implicit_rejection_set[:3]` for that
   `image_id` — not invented, not reordered against the pool's own priority.
2. `removed_answer_ar` (nota records) matches the pool's own
   `inferred_identity_ar` verbatim, and never appears in `options_ar`.
3. Control records: the real answer verbatim appears in `options_ar`, and
   `removed_answer_ar` is `null`.
4. `mcdr`/control NOTA option is always `"لا توجد إجابة صحيحة"`, always the
   last (4th) option — no per-item variation in this pass.
