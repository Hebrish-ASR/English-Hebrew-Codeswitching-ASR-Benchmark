"""ehcs-bench: run a model on the benchmark, or score transcripts produced elsewhere.

    ehcs-bench run   --model nemo:nvidia/parakeet-tdt-0.6b-v2 --name "Parakeet TDT 0.6B v2"
    ehcs-bench score --preds my_preds.jsonl --model-id my-org/my-model --name "My model"
    ehcs-bench rescore results/*.json      # recompute metrics from stored transcripts

A predictions file is JSONL, one {"id": ..., "hyp": ...} per line, covering every row of
the test set (code-switching and English control). The result file it writes is what the
leaderboard reads: see docs/submitting.md.
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

from . import data
from .score import score

SCHEMA = 1


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def build_result(rows, sha, hyps, model, extra=None):
    return {"schema": SCHEMA,
            "benchmark": {"repo": data.REPO, "version": data.VERSION, "revision": sha},
            "model": model, "scored_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **(extra or {}),
            "metrics": score(rows, hyps),
            "utterances": [{"id": r["id"], "subset": r["subset"], "ref": r["text"], "hyp": hyps.get(r["id"], "")} for r in rows]}


def model_meta(a, repo=None, revision=None, adapter=None, settings=None):
    return {"id": a.model_id or repo, "revision": revision, "name": a.name or (a.model_id or repo),
            "adapter": adapter or "external", "base_model": a.base_model, "fine_tuned_for_code_switching": a.fine_tuned,
            "params_m": a.params_m, "url": a.url or (f"https://huggingface.co/{repo}" if repo else None),
            "settings": settings or {}, "notes": a.notes, "contamination": a.contamination}


def write(res, out):
    out = Path(out or f"results/{slug(res['model']['name'])}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    c = res["metrics"]["subsets"]["clean"]
    print(f"{res['model']['name']}: clean WER {c['wer']:.2%}, term recall {c['terms']['all']['recall']:.1%} "
          f"(unseen {c['terms']['unseen']['recall']:.1%}), English WER "
          f"{res['metrics']['subsets']['english_control']['wer']:.2%} -> {out}")
    return out


def cmd_run(a):
    from .adapters import ADAPTERS, parse_spec, resolve_revision
    adapter, repo, rev = parse_spec(a.model)
    rev = resolve_revision(repo, rev)
    rows, _, sha = data.load(a.revision)
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time()
    texts, settings = ADAPTERS[adapter](repo, rev, [r["audio_path"] for r in rows], batch_size=a.batch_size)
    secs = time.time() - t0
    audio = sum(r["duration_s"] for r in rows)
    res = build_result(rows, sha, {r["id"]: t for r, t in zip(rows, texts)},
                       model_meta(a, repo, rev, adapter, settings),
                       {"run": {"wall_seconds": round(secs, 1), "audio_seconds": round(audio, 1), "host": a.host}})
    return write(res, a.out)


def cmd_score(a):
    rows, _, sha = data.load(a.revision)
    hyps = {}
    for l in Path(a.preds).read_text().splitlines():
        if l.strip():
            j = json.loads(l)
            hyps[j["id"]] = j["hyp"]
    res = build_result(rows, sha, hyps, model_meta(a))
    if res["metrics"]["missing"]:
        print(f"warning: {len(res['metrics']['missing'])} ids have no prediction and score as empty", file=sys.stderr)
    return write(res, a.out)


def cmd_rescore(a):
    rows, _, sha = data.load(a.revision)
    bad = 0
    for p in a.files:
        res = json.loads(Path(p).read_text())
        if res["benchmark"]["revision"] != sha:
            print(f"{p}: scored against {res['benchmark']['revision'][:12]}, benchmark is {sha[:12]}; skipping")
            bad += 1
            continue
        fresh = score(rows, {u["id"]: u["hyp"] for u in res["utterances"]})
        same = fresh["subsets"] == res["metrics"]["subsets"]
        print(f"{p}: {'OK' if same else 'METRICS DIFFER'}")
        bad += not same
        if a.write and not same:
            res["metrics"] = fresh
            Path(p).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ehcs-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--revision", default=data.REVISION, help="benchmark dataset revision")
        p.add_argument("--name", help="display name on the leaderboard")
        p.add_argument("--model-id", help="HF repo id or other stable identifier")
        p.add_argument("--base-model", help="what it was fine-tuned from, if anything")
        p.add_argument("--fine-tuned", action="store_true", help="fine-tuned for English-Hebrew code-switching")
        p.add_argument("--params-m", type=float, help="parameters, millions")
        p.add_argument("--url")
        p.add_argument("--notes")
        p.add_argument("--contamination", help="known overlap between the model's training data and this test set")
        p.add_argument("--out")

    r = sub.add_parser("run", help="transcribe the test set with a model, then score")
    common(r)
    r.add_argument("--model", required=True, help="<adapter>:<repo>[@rev], e.g. nemo:nvidia/parakeet-tdt-0.6b-v2")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--limit", type=int, help="first N rows only (smoke test; not a valid result)")
    r.add_argument("--host", default="local")
    s = sub.add_parser("score", help="score a predictions JSONL produced elsewhere")
    common(s)
    s.add_argument("--preds", required=True)
    v = sub.add_parser("rescore", help="recompute metrics from the transcripts in result files")
    v.add_argument("files", nargs="+")
    v.add_argument("--revision", default=data.REVISION)
    v.add_argument("--write", action="store_true", help="overwrite stale metrics")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "score":
        cmd_score(a)
    else:
        sys.exit(1 if cmd_rescore(a) else 0)


if __name__ == "__main__":
    main()
