## Inputs given per image

- The image itself (open and actually look at it — do not answer from the
  filename or metadata alone).
- Metadata **hints only**, which may be wrong and must not be trusted
  blindly: `subject`, `ground_truth_subject`, `category`, `country`,
  `region`, `medium`, `source`. Known failure modes in this metadata: raw
  Wikimedia filenames used as `subject` (e.g. `"File:Arabian Hot Dog.JPG"`),
  `ground_truth_subject: "UNMATCHED"`, missing `country`/`region`.

## Output: exactly these 4 fields, valid JSON, no extra keys

```json
{
  "inferred_identity_ar": "...",
  "inferred_identity_en": "...",
  "caption_ar": "...",
  "h_item_ar": "..."
}
```

### `inferred_identity_en` — the correct specific identity, in English

The most specific name you can confidently support from the image itself.
- If it's a specific named, recognizable landmark/artifact/dish/script
  style and you're confident, name it specifically (e.g. `"Al Faisaliah
  Tower"`, `"Kabsa"`, `"Naskh script"`).
- If the image only supports a generic-but-accurate identity, use that
  instead of guessing a specific name (e.g. `"modern glass skyscraper"`
  rather than inventing a building name you can't actually confirm from
  the picture).
- Cross-check against `ground_truth_subject`/`subject`, but do not defer to
  them if they look wrong (raw filename, `UNMATCHED`, or contradicted by
  what's visible) — correct or generalize instead.
- Never include information not inferable from the image (no invented
  dates, architects, restaurants, precise locations) even if the metadata
  hints at it — only carry forward metadata you can visually corroborate.

### `inferred_identity_ar` — Arabic translation of `inferred_identity_en`

Standard Arabic name/term for the same identity (not a transliteration,
unless the item has no common Arabic name, e.g. a proper noun with no
standard Arabic form). Modern Standard Arabic, no diacritics required.

### `caption_ar` — neutral, detailed, purely visual Arabic description

3–6 sentences, Modern Standard Arabic, describing only what is visible:
shape, color, material/texture, composition, layout, camera angle,
setting/background, distinguishing visual motifs. Style: formal, objective,
descriptive — like an image-recognition test item, not a tourist blurb (no
subjective praise, no historical trivia not visible in the frame).

**Hard rule: must NOT name or paraphrase-reveal the specific identity.**
Do not write the landmark/dish/artifact's proper name or a paraphrase that
gives it away (e.g. don't write "the famous tower with the golden orb" for
Al Faisaliah — describe the orb and the tapering silhouette, don't call it
"famous" or hint that it's iconic). This caption must remain usable later
to generate base-mode VQA questions without leaking the answer, and to
sanity-check the manifest label independent of naming it.

Worked example (already in `image_pool.json`, architecture/Al Faisaliah
Tower — read it before writing your own, this is the target style):

> صورة جوية لبرج شاهق يضيق تدريجيًا نحو القمة على شكل هرمي مدبب، بواجهة
> معدنية فضية مكسوة بنقش مثلثات متكررة وخطوط أفقية بارزة، وتعلوه كرة كبيرة
> داكنة اللون معلّقة عند الرأس المدبب مباشرة. التُقطت الصورة من زاوية
> مرتفعة تطل على أفق مدينة كثيفة المباني يكسوها ضباب أو غبار خفيف، وتظهر في
> الأسفل يسارًا قبة مبنى فاتحة اللون.

### `h_item_ar` — a plausible-but-WRONG Arabic term (foil), not the type/category

Per `image_schema.json`: *"Plausible-but-wrong Arabic term for this image,
chosen/generated fresh during the captioning pass... feeds hitem_ar in
base/sec/icc, and one entry in the pool nota samples from."*

This is a distractor, not a category label. Pick a **different, specific,
real, plausible item from the same domain** — one a model could
believably confuse with the correct answer, or that a misleading Arabic
caption could plausibly claim this image shows instead (Mode 2/3: cause-II
multimodal-inconsistency testing; Mode 5: NOTA distractor pool).

Guidelines:
- Same category as the image (architecture↔architecture, cuisine↔cuisine,
  attire↔attire, script↔script, objects↔objects).
- Specific, not generic (a named foil, not "a tower" or "a dish").
- Genuinely plausible as a mix-up — visually or culturally similar enough
  that stating it wouldn't be an obviously absurd claim (e.g. confusing two
  real Riyadh towers is plausible; confusing a tower with a dessert is
  not).
- Must be different from `inferred_identity_ar` (never equal).
- Real-world term (a real dish/landmark/script-style/garment/object name),
  not a nonsense placeholder.

Example: image is Al Faisaliah Tower (Riyadh) → `h_item_ar` = "برج
المملكة" (Kingdom Centre Tower) — a different real Riyadh landmark tower,
plausible enough to misattribute.

## Self-check before returning output

1. Did I actually look at the image, or am I echoing the metadata hints?
2. Does `caption_ar` avoid naming/revealing the identity?
3. Is `h_item_ar` a specific plausible foil, not a repeat of the identity
   and not a vague category word?
4. Is everything in Modern Standard Arabic (no dialect, no English mixed
   in except where a term has no Arabic equivalent)?
