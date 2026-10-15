"""Metrics. Pure Python plus jiwer; no GPU, no model. Definitions: docs/about.md and docs/data-dictionary.md.

The normalisation and the term matcher are the same as the ones behind the Parakeet
code-switching fine-tune's published numbers, so the two stay comparable.
"""
import re

SUBSET_GROUPS = {
    # name -> subsets it pools. `clean` is the leaderboard headline.
    "clean": ["clean"],
    "all_code_switch": ["clean", "contaminated_text", "contaminated_audio", "desktop_mic"],
    "english_control": ["english_control"],
}


def norm(t):
    """Lowercase, hyphens to spaces, drop punctuation except in-word apostrophes."""
    t = (t or "").lower().replace("’", "'").replace("-", " ")
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    return " ".join(t.split())


def term_re(term):
    """Word boundary, case-insensitive; hyphen and space interchangeable. Never substring."""
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term.strip())]
    return re.compile(r"(?<![\w])" + r"[\s-]+".join(parts) + r"(?![\w])", re.IGNORECASE)


def _wer(refs, hyps):
    import jiwer
    if not refs:
        return None, None
    hyps = [h or "<empty>" for h in hyps]
    w = jiwer.wer(refs, hyps)
    c = jiwer.cer(refs, hyps)
    return round(w, 4), round(c, 4)


def term_scores(rows, hyps):
    """Per-term recall over every Hebrew term occurrence in the references.

    strict    the canonical spelling appears in the hypothesis (any case)
    lenient   the canonical spelling or a lexicon variant appears (tzaharon / tsaharon)
    exact     the canonical spelling with the casing the reference uses (Bituach Leumi, not
              bituach leumi). The reference, not the term label, sets the case: in 7 clean
              occurrences the two differ (label Shacharit, sentence-medial shacharit).
    """
    per = {}
    for r, y in zip(rows, hyps):
        for t in r["hebrew_terms"]:
            d = per.setdefault(t["term"], {"n": 0, "strict": 0, "lenient": 0, "exact": 0, "unseen": t["unseen"]})
            d["n"] += 1
            rx = term_re(t["term"])
            m = rx.search(y or "")
            if m:
                d["strict"] += 1
                want = rx.search(r["text"]).group(0)
                d["exact"] += m.group(0).replace("-", " ") == want.replace("-", " ")
            if m or any(term_re(v).search(y or "") for v in t["variants"]):
                d["lenient"] += 1

    def agg(keep):
        sel = [d for d in per.values() if keep(d)]
        n = sum(d["n"] for d in sel)
        r = lambda k: round(sum(d[k] for d in sel) / n, 4) if n else None
        return {"occurrences": n, "terms": len(sel), "recall": r("strict"), "recall_lenient": r("lenient"),
                "recall_exact_case": r("exact")}

    return {"all": agg(lambda d: True), "unseen": agg(lambda d: d["unseen"]),
            "seen": agg(lambda d: not d["unseen"])}, per


def score(rows, hyps_by_id):
    """rows: benchmark metadata. hyps_by_id: {id: hypothesis}. Missing ids score as empty."""
    out = {"missing": sorted(r["id"] for r in rows if r["id"] not in hyps_by_id), "subsets": {}}
    for name, members in SUBSET_GROUPS.items():
        sel = [r for r in rows if r["subset"] in members]
        hyps = [hyps_by_id.get(r["id"], "") for r in sel]
        w, c = _wer([norm(r["text"]) for r in sel], [norm(h) for h in hyps])
        w_raw, _ = _wer([r["text"] for r in sel], hyps)
        s = {"n": len(sel), "minutes": round(sum(r["duration_s"] for r in sel) / 60, 1),
             "wer": w, "cer": c, "wer_raw": w_raw}
        if name != "english_control":
            terms, per = term_scores(sel, hyps)
            s["terms"] = terms
            if name == "clean":
                out["per_term"] = per
        out["subsets"][name] = s
    return out
