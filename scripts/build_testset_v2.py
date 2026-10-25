# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25", "soundfile", "scipy", "numpy"]
# ///
"""Build test set v2: domain-balanced code-switching clips, plus English and tech companions.

Every input is pinned, so a rerun gives the same files. Output: `data/v2/` of the HF dataset
Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark (v1 stays in `data/test/`). Method and
rationale: docs/about.md, "Test set v2".

    with-secrets uv run scripts/build_testset_v2.py --out build/v2 \
        [--manifests DIR]     # *.train.jsonl / *.val.jsonl from Modal parakeet-cs-finetune runs/<run>/
    hf upload Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark build/v2 . --repo-type dataset

The source dataset is gated: HF_TOKEN must be able to read it. The training manifests are not
public. With --manifests the trained-on exclusions are recomputed and written to
scripts/inputs/v2-trained-exclusions.json; without it, that committed file is used, so anyone
can rebuild the same set.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RAV = "danielrosehill/Read-Aloud-Voice-Sentences"
RAV_SHA = "2cec2ebc85e9a98aa7251051a8923e9c711c8dda"   # the revision Sabich v2.1 (run r10-all) trained on
EHMS = "Hebrish/English-Hebrew-Mixed-Sentences"        # Whisper-Hebrish's training data, as in v1
EHMS_SHA = "e7dc91ef7301"
# Domains: hebrish.app's lexicon, frozen in this repo (its source repo is private).
LEXICON = HERE / "inputs" / "hebrish-lexicon-2026-10-20.json"
# Spelling variants, as in v1, from a newer pinned commit of the public token classifier.
VARIANTS = ("https://raw.githubusercontent.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier/"
            "0459b004a7750bcd1a3531c90768514858cbb0fd/data/terms.csv")
# Variant spellings in source transcripts whose label is the canonical term; v1 found the first two
# (2026-10-15), v2 the third. Merged into the variants: rewritten here, accepted by lenient recall.
EXTRA_VARIANTS = {"cheder madregot": ["heder madregot"], "mahane yehuda": ["machane yehuda"],
                  "nahal oz": ["nachal oz"]}
EXCLUSIONS = HERE / "inputs" / "v2-trained-exclusions.json"
HF = "https://huggingface.co/datasets"

SEED = "ehcs-v2"
PER_DOMAIN = 10          # target clips per domain (fewer where the pool is short)
N_ENGLISH = 25
N_TECH = 25
OWNER = "Daniel Rosehill"
SR = 16000
HEBREW_CATS = {"code_switch", "names", "yiddish", "religious"}      # ASR-Training cs_lib.HEBREW_CATS
TECH_CATS = {"tech", "ai_interactions", "linux", "smart_home"}      # ASR-Training cs_lib.TECH_CATS
# Takes recorded for a special task: unusual delivery or a non-standard reference convention.
TASK_CATS = {"nato-alphabet", "singing", "disfluency", "whispered", "weird-voices", "repetitions",
             "long", "numeric"}
ENGLISH_CATS = {"general", "email", "message", "medical", "target_word", "homophone"}
# Reported first in tables; any other lexicon domain follows alphabetically.
RUNS_OF_RECORD = {"r10-all": "Sabich v2.1", "r9-all": "Sabich v2.0", "r7-all": "Sabich v1.0",
                  "r6-en30": "Jachnun v1.0"}


def hf_get(url):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok and url.startswith("https://huggingface.co/"):
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def jsonl(data):
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()]


def key(t):
    """Normalised text for duplicate and contamination checks (same as v1)."""
    return " ".join(re.sub(r"[^\w\s']", " ", (t or "").lower().replace("-", " ")).split())


def term_re(term):
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term.strip())]
    return re.compile(r"(?<![\w])" + r"[\s-]+".join(parts) + r"(?![\w])", re.IGNORECASE)


def rank(i):
    return hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest()


def is_owner(r):
    return (r.get("speaker") or OWNER) == OWNER


def has_transcript(r):
    return bool((r.get("text") or "").strip()) and r.get("transcript_status") != "pending"


def trained_exclusions(test, manifests):
    """Test-split ids whose id or normalised text is in any training/validation manifest."""
    per_run, by_id = {}, {}
    test_keys = {r["id"]: key(r["text"]) for r in test if has_transcript(r)}
    for p in sorted(Path(manifests).glob("*.jsonl")):
        run, part = p.name.rsplit(".", 2)[:2]
        rows = jsonl(p.read_bytes())
        ids, texts = {r.get("id") for r in rows}, {key(r.get("text")) for r in rows}
        hit_id = sorted(i for i in test_keys if i in ids)
        hit_text = sorted(i for i, k in test_keys.items() if k in texts and i not in ids)
        d = per_run.setdefault(run, {"train_rows": 0, "val_rows": 0, "id_matches": [], "text_matches": []})
        d[f"{part}_rows"] = len(rows)
        d["id_matches"] = sorted(set(d["id_matches"]) | set(hit_id))
        d["text_matches"] = sorted(set(d["text_matches"]) | set(hit_text))
        for i in hit_id:
            by_id.setdefault(i, set()).add(f"{run}:{part}:id")
        for i in hit_text:
            by_id.setdefault(i, set()).add(f"{run}:{part}:text")
    return {"source": f"{RAV}@{RAV_SHA[:12]} test split vs Modal volume parakeet-cs-finetune runs/<run>/{{train,val}}.jsonl",
            "checked": "2026-10-25", "runs": per_run,
            "excluded": {i: sorted(v) for i, v in sorted(by_id.items())}}


def to_16k(src_bytes, dst):
    """16 kHz mono 16-bit WAV, like ASR-Training training/cs_lib.py to_16k_wav. Returns (seconds, source format)."""
    from math import gcd
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(io.BytesIO(src_bytes))
    fmt = f"{info.format.lower()} {info.samplerate} Hz {info.subtype}"
    if (info.samplerate, info.channels, info.format, info.subtype) == (SR, 1, "WAV", "PCM_16"):
        dst.write_bytes(src_bytes)
        return round(info.duration, 3), fmt
    a, sr = sf.read(io.BytesIO(src_bytes), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype("float32")
    sf.write(str(dst), a, SR, subtype="PCM_16")
    return round(len(a) / SR, 3), fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/v2")
    ap.add_argument("--manifests", help="dir of <run>.train.jsonl / <run>.val.jsonl; recomputes the exclusions")
    ap.add_argument("--no-audio", action="store_true", help="metadata and report only")
    a = ap.parse_args()

    base = f"{HF}/{RAV}/resolve/{RAV_SHA}/data"
    test = jsonl(hf_get(f"{base}/test/metadata.jsonl"))
    train = jsonl(hf_get(f"{base}/train/metadata.jsonl"))
    ehms = []
    for s in ("whisper_train", "whisper_validation"):
        ehms += jsonl(hf_get(f"{HF}/{EHMS}/resolve/{EHMS_SHA}/{s}.jsonl"))
    ehms_text = {key(r["text"]) for r in ehms}
    train_terms = {t.lower() for r in train for t in (r.get("target_terms") or [])}

    lex = json.loads(LEXICON.read_text())
    domain_of = {t["t"].lower(): t["d"] for t in lex["terms"]}
    canon = {t["t"].lower(): t["t"] for t in lex["terms"]}
    variants = {}
    for row in csv.DictReader(io.StringIO(hf_get(VARIANTS).decode())):
        v = [x for x in (row.get("variants") or "").split("|") if x]
        if v:
            variants[row["term"].lower()] = v
    for k, v in EXTRA_VARIANTS.items():
        variants[k] = sorted(set(variants.get(k, [])) | set(v))

    if a.manifests:
        exc = trained_exclusions(test, a.manifests)
        EXCLUSIONS.write_text(json.dumps(exc, ensure_ascii=False, indent=1) + "\n")
    else:
        exc = json.loads(EXCLUSIONS.read_text())
    trained = set(exc["excluded"])

    # ---- eligibility ------------------------------------------------------------------------
    report = Counter()
    pool = []                       # rows that pass the common filters
    for r in sorted(test, key=lambda r: r["id"]):
        if not is_owner(r):
            report["drop_other_speaker"] += 1
        elif not has_transcript(r):
            report["drop_no_transcript"] += 1
        elif r.get("mode") == "import":
            report["drop_2025_desktop_import"] += 1
        elif r["id"] in trained:
            report["drop_trained_on"] += 1
        elif key(r["text"]) in ehms_text:
            report["drop_whisper_hebrish_text"] += 1
        elif TASK_CATS & set(r["categories"]):
            report["drop_task_take"] += 1
        else:
            pool.append(r)
    # One take per sentence: the lowest rank wins.
    seen, dedup = {}, []
    for r in sorted(pool, key=lambda r: rank(r["id"])):
        k = key(r["text"])
        if k in seen:
            report["drop_duplicate_sentence"] += 1
            continue
        seen[k] = r["id"]
        dedup.append(r)
    pool = sorted(dedup, key=lambda r: r["id"])

    def label(r):
        """Lexicon terms among the take's own target_terms, canonicalised into the reference."""
        text, terms, fixed, dropped = r["text"], [], [], []
        for t in r.get("target_terms") or []:
            lt = t.lower()
            if lt not in domain_of:
                dropped.append(t)          # English, tech, product, or a Hebrew word the lexicon lacks
                continue
            term = canon[lt]
            if any(x["term"] == term for x in terms):
                continue
            if not term_re(term).search(text):
                for v in variants.get(lt, []):
                    m = term_re(v).search(text)
                    if m:
                        text = text[:m.start()] + term + text[m.end():]
                        fixed.append(f"{m.group(0)} -> {term}")
                        break
                else:
                    dropped.append(t + " (not in text)")
                    continue
            terms.append({"term": term, "variants": variants.get(lt, []), "unseen": lt not in train_terms,
                          "domain": domain_of[lt]})
        return text, terms, fixed, dropped

    # ---- headline: code-switching, one domain per clip --------------------------------------
    cands, no_lex, tech_cs, dropped_labels = [], 0, 0, Counter()
    for r in pool:
        cats = set(r["categories"])
        if "code_switch" not in cats:
            continue
        if cats & TECH_CATS:
            tech_cs += 1                   # code-switching and tech are kept apart (Daniel, 2026-10-24)
            continue
        text, terms, fixed, dropped = label(r)
        dropped_labels.update(dropped)
        if not terms:
            no_lex += 1
            continue
        cands.append({"r": r, "text": text, "terms": terms, "fixed": fixed})
    report["cs_dropped_tech"] = tech_cs
    report["cs_dropped_no_lexicon_term"] = no_lex

    # Domain rule: majority domain of the clip's terms, with `people` counted only when the clip has
    # nothing else (names are mostly addressees: "Hi Gal, ..."). Ties go to the domain with fewer
    # candidate clips, then alphabetically.
    def doms(c):
        ds = [t["domain"] for t in c["terms"]]
        return [d for d in ds if d != "people"] or ds
    supply = Counter(d for c in cands for d in set(doms(c)))
    for c in cands:
        n = Counter(doms(c))
        c["domain"] = min(n, key=lambda d: (-n[d], supply[d], d))
    by_dom = defaultdict(list)
    for c in cands:
        by_dom[c["domain"]].append(c)

    headline, shortfall = [], {}
    for d in sorted(lex["domains"]):
        cs = sorted(by_dom.get(d, []), key=lambda c: rank(c["r"]["id"]))
        # First pass: clips that add a term not yet chosen in this domain; then fill in rank order.
        chosen, have = [], set()
        for c in cs:
            if len(chosen) < PER_DOMAIN and not {t["term"] for t in c["terms"]} <= have:
                chosen.append(c)
                have |= {t["term"] for t in c["terms"]}
        for c in cs:
            if len(chosen) < PER_DOMAIN and c not in chosen:
                chosen.append(c)
        if len(chosen) < PER_DOMAIN:
            shortfall[d] = {"available": len(cs), "target": PER_DOMAIN}
        headline += chosen

    # ---- companions -------------------------------------------------------------------------
    hebrew_free = lambda r: not (set(r["categories"]) & (HEBREW_CATS | TECH_CATS)) and not any(
        term_re(t).search(r["text"]) for t in (r.get("target_terms") or []) if t.lower() in domain_of)
    english = [r for r in pool if hebrew_free(r) and set(r["categories"]) <= ENGLISH_CATS]
    english = sorted(english, key=lambda r: rank(r["id"]))[:N_ENGLISH]
    tech = [r for r in pool if set(r["categories"]) & TECH_CATS and not set(r["categories"]) & HEBREW_CATS]
    tech = sorted(tech, key=lambda r: rank(r["id"]))[:N_TECH]

    # ---- write ------------------------------------------------------------------------------
    out = Path(a.out) / "data" / "v2"
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True)
    rows = []

    def emit(r, subset, text, terms, domain, fixed):
        fn = f"audio/{r['id']}.wav"
        dur, fmt = r["duration_s"], None
        if not a.no_audio:
            dur, fmt = to_16k(hf_get(f"{base}/test/{r['file_name']}"), out / fn)
        rows.append({"file_name": fn, "id": r["id"], "text": text, "subset": subset, "domain": domain,
                     "duration_s": dur, "hebrew_terms": terms, "categories": r["categories"],
                     "recording": "phone_2026", "source_audio": fmt,
                     "source_dataset": f"{RAV}@{RAV_SHA[:12]}", "source_text": r["text"] if fixed else None})
        print(r["id"], subset, domain or "", flush=True)

    for c in sorted(headline, key=lambda c: c["r"]["id"]):
        emit(c["r"], "clean", c["text"], c["terms"], c["domain"], c["fixed"])
    for r in sorted(english, key=lambda r: r["id"]):
        emit(r, "english_control", r["text"], [], None, [])
    for r in sorted(tech, key=lambda r: r["id"]):
        emit(r, "tech", r["text"], [], None, [])
    rows.sort(key=lambda x: x["id"])
    (out / "metadata.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))

    clean = [x for x in rows if x["subset"] == "clean"]
    occ = [t for x in clean for t in x["hebrew_terms"]]
    rep = {
        "source": f"{RAV}@{RAV_SHA}", "seed": SEED, "per_domain_target": PER_DOMAIN,
        "test_split_rows": len(test), "filters": dict(report), "eligible_pool": len(pool),
        "cs_candidates_by_domain": {d: len(by_dom.get(d, [])) for d in sorted(lex["domains"])},
        "clean_by_domain": dict(sorted(Counter(x["domain"] for x in clean).items())),
        "shortfall": shortfall,
        "subsets": {s: {"clips": sum(x["subset"] == s for x in rows),
                        "minutes": round(sum(x["duration_s"] for x in rows if x["subset"] == s) / 60, 1)}
                    for s in ("clean", "english_control", "tech")},
        "clean_terms": {"occurrences": len(occ), "distinct": len({t["term"] for t in occ}),
                        "unseen_occurrences": sum(t["unseen"] for t in occ)},
        "canonicalised": {x["id"]: x["source_text"] for x in rows if x["source_text"]},
        "target_terms_not_labelled": dict(dropped_labels.most_common()),
        "trained_on": {"excluded_test_ids": len(trained),
                       "runs": {k: {"label": RUNS_OF_RECORD.get(k), "id": len(v["id_matches"]),
                                    "text_only": len(v["text_matches"])} for k, v in exc["runs"].items()}},
    }
    (out / "build-report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("filters", "clean_by_domain", "shortfall", "subsets", "clean_terms")}, indent=1))


if __name__ == "__main__":
    main()
