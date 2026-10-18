"""
Explanation analysis for the full Sarab run. Read-only: saved result CSVs, no API calls.

    python3 analysis/explanation_analysis.py   # writes analysis/results/explanation_analysis.md and contradiction_candidates.csv

Two measures, both rule-based and therefore approximate (precision of the contradiction rule is estimated by reading a sample):
  1. caption citation (Sway, False): how often the explanation refers to the given caption/description, split by right/wrong answers
  2. answer-explanation contradiction (Ground, Sway, False): the question asks about one entity (the true identity for a 'yes'
     question, the distractor for a 'no' question). A contradiction candidate is a reply whose first word says yes but whose
     explanation denies that entity, or says no but whose explanation affirms it.
"""
import csv, importlib.util, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_analysis as ma

ROOT = ma.ROOT; OUT = ma.OUT
spec = importlib.util.spec_from_file_location("m5", ROOT / "modes/mode5_nota/run_mode5_openrouter.py")
m5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m5)

CITE = re.compile(r"الوصف|وصف|السياق|النص المرفق|كما ورد|كما ذكر|كما في السؤال|كما وصف|حسب النص|المذكور في")
NEG_WORDS = {"ليس", "ليست", "وليس", "وليست", "ولا", "لا", "غير", "بدلا", "بخلاف", "بينما", "اما", "عن"}
IDENT_WORDS = {"هو", "هي", "تظهر", "يظهر", "الظاهر", "الظاهره", "عباره", "يبدو", "تبدو", "هذا", "هذه", "بل"}
STOP = {"هذا", "هذه", "صوره", "الصوره", "ظاهر", "الظاهر"}


def words(text):
    out = []
    for w in re.findall(r"[\u0621-\u064A]+", m5._norm_one(text)):
        out.append(w)
    return out


def base(w):
    return w[2:] if w.startswith("ال") and len(w) > 3 else w


def toks(t):
    return {x for x in m5._name_tokens(t) if x not in STOP}


def stance(text, entity):
    """(affirm, deny): does the explanation assert the asked entity as what the image shows (an identity word such as
    'هو/هي/تظهر/بل' just before it) or reject it (a negation or contrast word such as 'ليس/بينما/بخلاف' just before it)?
    A mention with neither cue (for example the explanation opening with the entity as its subject) counts as neutral."""
    E = toks(entity)
    if not E:
        return False, False
    ws = words(text); bs = [base(w) for w in ws]; aff = den = False
    for i, b in enumerate(bs):
        if b not in E:
            continue
        window = set(bs[max(0, i - 1): i + len(E) + 2])
        if len(window & E) / len(E) < 0.6:
            continue
        before = ws[max(0, i - 4): i]
        if any(w in NEG_WORDS for w in before[-3:]):
            den = True
        elif any(w in IDENT_WORDS for w in before):
            aff = True
    return aff, den


def main():
    data = ma.load(); md = []; cand = []; cite = {}
    md.append("## 1. Caption citation in Sway and False (600 replies per model)\n")
    md.append("A reply 'cites' the caption if its explanation contains a word such as وصف / السياق / كما ورد / كما في السؤال (keyword count, approximate).\n")
    md.append("| Model | Cites caption | Of wrong answers | Of right answers | Wrong answers that cite | Wrong that do not cite |"); md.append("|" + "---|" * 6)
    for m in ma.MODEL_ORDER:
        rows = [r for mode in (2, 3) for r in data[(m, mode)]]
        n = len(rows); cw = cr = w = rt = 0
        for r in rows:
            ok = ma.is_ok(r["ground_truth"], ma.lenient_parse(r)[1]); c = bool(CITE.search(r["raw_response"]))
            if ok: rt += 1; cr += c
            else: w += 1; cw += c
        md.append("| %s | %.0f%% | %.0f%% | %.0f%% | %d | %d |" % (ma.SHORT[m], 100 * (cw + cr) / n, 100 * cw / max(w, 1), 100 * cr / max(rt, 1), cw, w - cw))
        cite[m] = (cw, w, cr, rt)

    md.append("\n## 2. Answer-explanation contradiction candidates (Ground, Sway, False; 960 replies per model)\n")
    md.append("| Model | Rule candidates (UNVERIFIED, mostly false alarms) | Candidate rate | Of which the first word was scored wrong | Said no, explanation seems to affirm the entity | Said yes, explanation seems to deny it |"); md.append("|" + "---|" * 6)
    for m in ma.MODEL_ORDER:
        c_no_aff = c_yes_den = wrong = n = 0
        for mode in (1, 2, 3):
            for r in data[(m, mode)]:
                pred = ma.lenient_parse(r)[1]
                if pred not in ("yes", "no"):
                    continue
                n += 1
                entity = r["gt_ar"] if r["polarity"] == "yes" else r["h_item_ar"]
                text = re.sub(r"^[\s\"'*#\-–—.,:;!?()\[\]]*(?:نعم|لا)", "", r["raw_response"], count=1)   # drop the answer word
                aff, den = stance(text, entity)
                hit = None
                if pred == "no" and aff and not den: hit = "said no; explanation affirms"; c_no_aff += 1
                if pred == "yes" and den and not aff: hit = "said yes; explanation denies"; c_yes_den += 1
                if hit:
                    ok = ma.is_ok(r["ground_truth"], pred); wrong += (not ok)
                    cand.append([ma.SHORT[m], ma.MODES[mode], r["task_id"], r["polarity"], r["ground_truth"], pred, int(ok), hit,
                                 entity, r["raw_response"].replace("\n", " ")[:300]])
        tot = c_no_aff + c_yes_den
        md.append("| %s | %d | %.1f%% | %d | %d | %d |" % (ma.SHORT[m], tot, 100 * tot / n, wrong, c_no_aff, c_yes_den))

    labels = {}
    lf = Path(__file__).resolve().parent / "contradiction_labels.csv"
    if lf.is_file():
        labels = {(r["model"], r["task_id"]): r["label"] for r in csv.DictReader(open(lf, encoding="utf-8"))}
    with open(OUT / "contradiction_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "mode", "task_id", "polarity", "ground_truth", "parsed", "scored_correct", "type", "asked_entity", "reply", "verified (TRUE/FALSE)"])
        for row in cand: w.writerow(row + [labels.get((row[0], row[2]), "")])
    if labels:
        md.append("\n### Verified contradictions (every candidate was read; labels in `analysis/contradiction_labels.csv`)\n")
        md.append("A candidate is verified TRUE when the first word contradicts what the explanation says about the asked entity. "
                  "The rule alone is only about %.0f%% precise (%d of %d candidates), so the verified count below is the one to use. It is a **lower bound**: "
                  "contradictions the rule misses are not counted.\n" % (100 * sum(v == "TRUE" for v in labels.values()) / len(labels), sum(v == "TRUE" for v in labels.values()), len(labels)))
        md.append("| Model | Verified contradictions | Per 960 replies | Of which the first word was scored wrong |"); md.append("|" + "---|" * 4)
        for m in ma.MODEL_ORDER:
            mine = [r for r in cand if r[0] == ma.SHORT[m] and labels.get((r[0], r[2])) == "TRUE"]
            md.append("| %s | %d | %.1f%% | %d |" % (ma.SHORT[m], len(mine), 100 * len(mine) / 960, sum(1 for r in mine if not r[6])))
        md.append("\nPattern: in the verified cases the explanation usually follows the image while the first word follows the misleading caption (for example 'no' to a question whose entity the explanation then names as what the image shows).\n")
    md.append("\nAll %d candidates are in `analysis/results/contradiction_candidates.csv`. The labels were made in one pass by the assistant and should get a human check.\n" % len(cand))
    text = "\n".join(md)
    (OUT / "explanation_analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return cand


if __name__ == "__main__":
    main()
