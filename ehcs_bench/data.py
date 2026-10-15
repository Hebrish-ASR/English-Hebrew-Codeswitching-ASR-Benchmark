"""The frozen test set on the Hub. Pinned by revision so every result scores the same audio."""
import json
from pathlib import Path

REPO = "danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark"
VERSION = "v1"
# Commit of REPO that holds test-set v1. Results record this and the leaderboard refuses
# results scored against anything else. Bump VERSION and this together, never one alone.
REVISION = "3cb0b2b3b3e46c951962ababeb7c56fc4248cb94"


def load(revision=REVISION, local_dir=None):
    """Returns (rows, data_dir, resolved_sha). Rows carry an `audio_path`."""
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi().dataset_info(REPO, revision=revision).sha
    root = Path(snapshot_download(REPO, repo_type="dataset", revision=sha, allow_patterns=["data/test/*", "data/test/audio/*"],
                                  local_dir=local_dir))
    d = root / "data" / "test"
    rows = [json.loads(l) for l in (d / "metadata.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["audio_path"] = str(d / r["file_name"])
    return rows, d, sha
