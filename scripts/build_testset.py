"""Build the frozen benchmark test set from pinned upstream revisions.

Every input is pinned, so running this again produces the same files. The output is the
`data/test/` folder of the HF dataset Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark.
How each subset is decided, and why: docs/contamination.md.

    uv run scripts/build_testset.py --out build/
    hf upload Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark build/ . --repo-type dataset
"""
import argparse
import csv
import io
import json
import re
import urllib.request
from pathlib import Path

RAV = "danielrosehill/Read-Aloud-Voice-Sentences"
RAV_SHA = "3bdaa2a50d5246d222f8c719861ff738787d2811"   # the revision Parakeet CS v2 trained on
# English-Hebrew-Mixed-Sentences as it stood when Whisper-Hebrish was trained (2025-11-17).
# Later revisions (2026-10-14) appended phone takes, so `main` is the wrong thing to diff against.
EHMS = "Hebrish/English-Hebrew-Mixed-Sentences"
EHMS_SHA = "e7dc91ef7301"
LEXICON = "https://raw.githubusercontent.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier/94f4a72e6043468c731cfb22036fd3d306876f40/data/terms.csv"

HF = "https://huggingface.co/datasets"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def jsonl(url):
    return [json.loads(l) for l in get(url).decode().splitlines() if l.strip()]


def key(t):
    return " ".join(re.sub(r"[^\w\s']", " ", t.lower().replace("-", " ")).split())


def term_re(term):
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term.strip())]
    return re.compile(r"(?<![\w])" + r"[\s-]+".join(parts) + r"(?![\w])", re.IGNORECASE)


# Variant spellings left in source transcripts whose label is the canonical term (found
# 2026-10-15: the source's canonicalisation pass missed these). Merged into the lexicon's
# variants, so they are both rewritten here and accepted by the lenient term metric.
EXTRA_VARIANTS = {"cheder madregot": ["heder madregot"], "mahane yehuda": ["machane yehuda"]}


def canonicalise(text, terms):
    """A labelled term must appear in its reference, or no transcript can score a hit on it.
    Where the reference carries a variant instead, rewrite that span to the canonical term."""
    fixed = []
    for t in terms:
        if term_re(t["term"]).search(text):
            continue
        for v in t["variants"]:
            m = term_re(v).search(text)
            if m:
                text = text[:m.start()] + t["term"] + text[m.end():]
                fixed.append(f"{m.group(0)} -> {t['term']}")
                break
        else:
            raise SystemExit(f"term {t['term']!r} not in reference {text!r}")
    return text, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    out = Path(a.out) / "data" / "test"
    (out / "audio").mkdir(parents=True, exist_ok=True)

    test = jsonl(f"{HF}/{RAV}/resolve/{RAV_SHA}/data/test/metadata.jsonl")
    train = jsonl(f"{HF}/{RAV}/resolve/{RAV_SHA}/data/train/metadata.jsonl")
    ehms_train = []
    for s in ("whisper_train", "whisper_validation"):
        ehms_train += jsonl(f"{HF}/{EHMS}/resolve/{EHMS_SHA}/{s}.jsonl")
    ehms_ids = {r["id"] for r in ehms_train}
    ehms_text = {key(r["text"]) for r in ehms_train}
    train_terms = {t.lower() for r in train for t in r["target_terms"]}

    variants = {}
    for row in csv.DictReader(io.StringIO(get(LEXICON).decode())):
        v = [x for x in (row.get("variants") or "").split("|") if x]
        if v:
            variants[row["term"].lower()] = v
    for k, v in EXTRA_VARIANTS.items():
        variants[k] = sorted(set(variants.get(k, [])) | set(v))

    rows, counts = [], {}
    for r in sorted(test, key=lambda r: r["id"]):
        cs = "code_switch" in r["categories"]
        if not cs:
            subset = "english_control"
        elif r.get("mode") == "import" and r["id"].removeprefix("ehms_") in ehms_ids:
            subset = "contaminated_audio"      # this exact recording is in Whisper-Hebrish's training set
        elif r.get("mode") == "import":
            subset = "desktop_mic"             # older desktop-mic recording, not trained on
        elif key(r["text"]) in ehms_text:
            subset = "contaminated_text"       # new recording of a sentence Whisper-Hebrish trained on
        else:
            subset = "clean"
        counts[subset] = counts.get(subset, 0) + 1
        terms = [{"term": t, "variants": variants.get(t.lower(), []), "unseen": t.lower() not in train_terms}
                 for t in r["target_terms"]] if cs else []
        text, fixed = canonicalise(r["text"], terms)
        if fixed:
            print(f"  canonicalised {r['id']}: {fixed}", flush=True)
        src = f"{HF}/{RAV}/resolve/{RAV_SHA}/data/test/{r['file_name']}"
        (out / r["file_name"]).write_bytes(get(src))
        rows.append({"file_name": r["file_name"], "id": r["id"], "text": text, "subset": subset,
                     "duration_s": r["duration_s"], "hebrew_terms": terms, "categories": r["categories"],
                     "recording": "desktop_mic_2025" if r.get("mode") == "import" else "phone_2026",
                     "source_dataset": f"{RAV}@{RAV_SHA[:12]}", "source_text": r["text"] if fixed else None})
        print(r["id"], subset, flush=True)

    (out / "metadata.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
