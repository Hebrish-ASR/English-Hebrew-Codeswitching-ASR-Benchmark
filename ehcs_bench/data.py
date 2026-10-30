"""The frozen test sets on the Hub. Pinned by revision so every result scores the same audio.

Each version lives in its own folder of REPO and is pinned to the commit that added it.
v1 (2026-10-15): data/test/, 145 clips. v2 (2026-10-25): data/v2/, domain-balanced, the current set.

The pinned commits are HF audiofolders (<folder>/metadata.jsonl + <folder>/audio/). On 2026-10-30 `main` was
converted to Parquet (Voice-Training-Recorder docs/parquet-layout.md): splits `test` (v1) and `v2` as
data/<split>-*.parquet shards, index meta/<split>.jsonl with each take's byte range in its shard. Earlier
commits keep the audiofolder, so the pins are unchanged; load() reads either layout, by the commit asked for.
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
    api = HfApi()
    sha = api.dataset_info(REPO, revision=revision).sha
    if api.file_exists(REPO, "meta/schema.json", repo_type="dataset", revision=sha):
        return (*_load_parquet(folder.split("/")[-1], sha, local_dir), sha)
    root = Path(snapshot_download(REPO, repo_type="dataset", revision=sha,
                                  allow_patterns=[f"{folder}/metadata.jsonl", f"{folder}/audio/*"], local_dir=local_dir))
    d = root / folder
    rows = [json.loads(l) for l in (d / "metadata.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["audio_path"] = str(d / r["file_name"])
    return rows, d, sha


def _load_parquet(split, sha, local_dir=None):
    """Rows of `split` at a Parquet commit, each take's audio file cut out of its shard by its byte range
    (audio is stored verbatim, uncompressed) into <local_dir or a temp dir>/<sha12>/<split>/<audio_path>."""
    import tempfile
    from huggingface_hub import hf_hub_download
    idx = hf_hub_download(REPO, f"meta/{split}.jsonl", repo_type="dataset", revision=sha)
    rows = [json.loads(l) for l in Path(idx).read_text().splitlines() if l.strip()]
    out = Path(local_dir or tempfile.gettempdir()) / f"ehcs-{sha[:12]}" / split
    shards = {}
    for r in rows:
        dest = out / r["audio_path"]
        if not dest.exists():
            if r["shard"] not in shards:
                shards[r["shard"]] = hf_hub_download(REPO, r["shard"], repo_type="dataset", revision=sha)
            with open(shards[r["shard"]], "rb") as f:
                f.seek(r["audio_offset"])
                data = f.read(r["audio_length"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        r["file_name"], r["audio_path"] = r["audio_path"], str(dest)
    return rows, out
