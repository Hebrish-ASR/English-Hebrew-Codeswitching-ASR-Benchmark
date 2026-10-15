#!/usr/bin/env bash
# Assemble the leaderboard Space from space/, results/ and the pinned test-set metadata,
# then upload it. The Space reads only these files; it never calls the Hub at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard
REV=$(python3 -c "import re;print(re.search(r'^REVISION = \"(\w+)\"', open('ehcs_bench/data.py').read(), re.M).group(1))")
STAGE=$(mktemp -d)
cp space/app.py space/README.md space/requirements.txt "$STAGE"/
cp docs/about.md "$STAGE"/ABOUT.md
mkdir "$STAGE"/results && cp results/*.json "$STAGE"/results/
printf '{"revision": "%s"}\n' "$REV" > "$STAGE"/benchmark.json
curl -sfL "https://huggingface.co/datasets/danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark/resolve/$REV/data/test/metadata.jsonl" -o "$STAGE"/metadata.jsonl
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
HF=${HF_CLI:-$HOME/.local/bin/hf}
"$HF" repos create "$SPACE" --repo-type space --space-sdk gradio --exist-ok >/dev/null
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --delete "results/*" --commit-message "${1:-Deploy leaderboard}"
rm -rf "$STAGE"
