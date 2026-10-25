#!/usr/bin/env bash
# Assemble the leaderboard Space from space/, results/, archive/, the docs and the pinned test-set metadata,
# then upload it. The Space reads only these files; it never calls the Hub at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard
# Every test set in ehcs_bench/data.py VERSIONS gets a tab; the current VERSION is the default one.
read -r CUR VERS < <(python3 -c "import ehcs_bench.data as d; print(d.VERSION, ' '.join(d.VERSIONS))")
STAGE=$(mktemp -d)
cp space/app.py space/README.md space/requirements.txt "$STAGE"/
cp docs/about.md "$STAGE"/ABOUT.md
cp docs/data-dictionary.md "$STAGE"/DATA_DICTIONARY.md
mkdir "$STAGE"/results "$STAGE"/archive && cp archive/*.json "$STAGE"/archive/
BENCH_JSON=$(python3 -c "import json, ehcs_bench.data as d; print(json.dumps({'current': d.VERSION, 'revision': d.REVISION, 'versions': {v: s for v, (s, _) in d.VERSIONS.items()}}))")
echo "$BENCH_JSON" > "$STAGE"/benchmark.json
for V in $VERS; do
  read -r SHA DIR < <(python3 -c "import ehcs_bench.data as d; print(*d.VERSIONS['$V'])")   # DIR: data/test (v1), data/v2 (v2)
  # Repo layout: v1 results in results/*.json, later versions in results/<version>/. On the Space: results/<version>/.
  mkdir "$STAGE/results/$V" && cp "results/$([ "$V" = v1 ] || echo "$V/")"*.json "$STAGE/results/$V/"
  curl -sfL "https://huggingface.co/datasets/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark/resolve/$SHA/$DIR/metadata.jsonl" -o "$STAGE/metadata-$V.jsonl"
done
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
HF=${HF_CLI:-$HOME/.local/bin/hf}
"$HF" repos create "$SPACE" --repo-type space --space-sdk gradio --exist-ok >/dev/null
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --delete "results/**" --delete "archive/*" --delete "metadata.jsonl" --commit-message "${1:-Deploy leaderboard}"
rm -rf "$STAGE"
