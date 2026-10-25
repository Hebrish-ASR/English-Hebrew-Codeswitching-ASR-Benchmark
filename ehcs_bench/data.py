"""The frozen test sets on the Hub. Pinned by revision so every result scores the same audio.

Each version lives in its own folder of REPO and is pinned to the commit that added it.
v1 (2026-10-15): data/test/, 145 clips. v2 (2026-10-25): data/v2/, domain-balanced, the current set.
"""
import json
from pathlib import Path

REPO = "Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark"
# The current test set. Results record the version and commit; the leaderboard shows only
# results scored against REVISION. Bump VERSION and REVISION together, never one alone. Both stay
# string literals: scripts/*.py and deploy_space.sh read them with a regex.
VERSION = "v2"
REVISION = "48ec85a301dc593979adb701cb52f13bb66305d4"
# version -> (commit of REPO that holds it, folder in REPO). A new test set is a new entry,
# never an edit of an old one.
VERSIONS = {
    "v1": ("3cb0b2b3b3e46c951962ababeb7c56fc4248cb94", "data/test"),
    "v2": (REVISION, "data/v2"),
}


def version_of(revision):
    """The version pinned at `revision` (a full or abbreviated commit), else None."""
    return next((v for v, (sha, _) in VERSIONS.items() if revision and sha.startswith(revision)), None)


def load(revision=None, local_dir=None, version=None):
    """Returns (rows, data_dir, resolved_sha). Rows carry an `audio_path`.
    Pass a version ("v1", "v2") or a revision; with neither, the current VERSION."""
    from huggingface_hub import HfApi, snapshot_download
    version = version or version_of(revision) or VERSION
    revision = revision or VERSIONS[version][0]
    folder = VERSIONS[version][1]
    sha = HfApi().dataset_info(REPO, revision=revision).sha
    root = Path(snapshot_download(REPO, repo_type="dataset", revision=sha,
                                  allow_patterns=[f"{folder}/metadata.jsonl", f"{folder}/audio/*"], local_dir=local_dir))
    d = root / folder
    rows = [json.loads(l) for l in (d / "metadata.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["audio_path"] = str(d / r["file_name"])
    return rows, d, sha
