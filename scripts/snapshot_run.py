"""Freeze the current leaderboard as a dated benchmark round in archive/<date>.json.

    python3 scripts/snapshot_run.py --date 2026-10-15 --title "First round" --commit aa09470

results/ is "latest": re-running a model overwrites its file. archive/ is append-only: one
file per round, never edited after it is committed. Each model entry keeps the summary
metrics (no transcripts); the transcripts stay recoverable from `repo_commit`.
The Space's Archive tab reads these files; deploy_space.sh copies them.
"""
import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def summary(r):
    m, s = r["model"], r["metrics"]["subsets"]
    return {"name": m["name"], "id": m["id"], "revision": m.get("revision"), "url": m.get("url"),
            "fine_tuned_for_code_switching": m.get("fine_tuned_for_code_switching"), "params_m": m.get("params_m"),
            "contamination": m.get("contamination"), "scored_at": r["scored_at"], "host": r.get("run", {}).get("host"),
            "subsets": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--title", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--commit", help="repo commit that holds these results/ (default: HEAD)")
    a = ap.parse_args()
    rev = re.search(r'^REVISION = "(\w+)"', (ROOT / "ehcs_bench/data.py").read_text(), re.M).group(1)
    ver = re.search(r'^VERSION = "(\w+)"', (ROOT / "ehcs_bench/data.py").read_text(), re.M).group(1)
    commit = subprocess.check_output(["git", "rev-parse", a.commit or "HEAD"], cwd=ROOT, text=True).strip()
    res = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
    res = [r for r in res if r["benchmark"]["revision"] == rev]
    key = lambda r: (-r["metrics"]["subsets"]["clean"]["terms"]["all"]["recall"], r["metrics"]["subsets"]["clean"]["wer"])
    out = ROOT / "archive" / f"{a.date}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; archive rounds are immutable. Use another --date (e.g. {a.date}b).")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "schema": 1, "date": a.date, "title": a.title, "notes": a.notes,
        "benchmark": {"version": ver, "revision": rev}, "repo_commit": commit,
        "models": [summary(r) for r in sorted(res, key=key)]}, ensure_ascii=False, indent=1) + "\n")
    print(f"{out}: {len(res)} models, test set {ver}@{rev[:12]}, results at {commit[:12]}")


if __name__ == "__main__":
    main()
