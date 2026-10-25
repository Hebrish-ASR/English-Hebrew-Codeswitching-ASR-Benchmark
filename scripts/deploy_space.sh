#!/usr/bin/env bash
# Assemble the leaderboard Space from space/, results/, archive/, the docs and the pinned test-set metadata,
# then upload it. The Space reads only these files; it never calls the Hub at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard
REV=$(python3 -c "import ehcs_bench.data as d; print(d.REVISION)")
DIR=$(python3 -c "import ehcs_bench.data as d; print(d.VERSIONS[d.VERSION][1])")   # data/test (v1), data/v2 (v2)
STAGE=$(mktemp -d)
cp space/app.py space/README.md space/requirements.txt "$STAGE"/
cp docs/about.md "$STAGE"/ABOUT.md
cp docs/data-dictionary.md "$STAGE"/DATA_DICTIONARY.md
VER=$(python3 -c "import ehcs_bench.data as d; print(d.VERSION)")
# Only the current test set's results: the Space shows nothing else. v1: results/*.json; later: results/<version>/.
mkdir "$STAGE"/results && cp "results/$([ "$VER" = v1 ] || echo "$VER/")"*.json "$STAGE"/results/
mkdir "$STAGE"/archive && cp archive/*.json "$STAGE"/archive/
printf '{"revision": "%s"}\n' "$REV" > "$STAGE"/benchmark.json
curl -sfL "https://huggingface.co/datasets/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark/resolve/$REV/$DIR/metadata.jsonl" -o "$STAGE"/metadata.jsonl
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
HF=${HF_CLI:-$HOME/.local/bin/hf}
"$HF" repos create "$SPACE" --repo-type space --space-sdk gradio --exist-ok >/dev/null
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --delete "results/*" --delete "archive/*" --commit-message "${1:-Deploy leaderboard}"
rm -rf "$STAGE"
