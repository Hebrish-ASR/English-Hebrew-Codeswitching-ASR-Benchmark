# English-Hebrew Code-Switching ASR Benchmark

A replicable benchmark for speech recognition on **English speech with Hebrew words in it**: *"pick up some bamba from the makolet"*, *"I have to go to Bituach Leumi tomorrow"*. It scores stock models and code-switching fine-tunes on the same frozen test set, with the same normaliser and term matcher.

> **Status (2026-10-25): test set v2 is current.** v2 has 129 code-switching clips balanced across 13 Hebrew vocabulary domains, none of them trained on by any Hebrish model, plus English and tech checks. It is the set Hebrish uses for its own internal benchmarks; see [About](docs/about.md#test-set-v2-current-2026-10-25). The table below and the [leaderboard Space](https://huggingface.co/spaces/danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard) default to **v2** (first round scored 2026-10-25); v1 stays loadable (`--version v1`) and keeps its own tab on the Space with every result scored on it. v1's known issue (22 of its 139 labelled terms are English tech or product words) is fixed in v2. Still single-speaker read speech. Contributions welcome: more speakers, better labels, more models.

| | |
|---|---|
| **Website** (the Parakeet code-switching fine-tune) | [hebrish.app](https://hebrish.app): get the Hebrish keyboard for Android and the desktop apps, see the results, and find every model, dataset and demo in one place |
| **Leaderboard** | [HF Space: English-Hebrew-Codeswitching-ASR-Leaderboard](https://huggingface.co/spaces/danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard): test set v2 by default (with a per-domain view and tech WER), v1 in its own tab |
| **Test set** | [HF dataset: English-Hebrew-Codeswitching-ASR-Benchmark](https://huggingface.co/datasets/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark): v2 (`data/v2/`, current) and v1 (`data/test/`), pinned commits in [`ehcs_bench/data.py`](ehcs_bench/data.py) |
| **Methodology, contamination, limits** | [`docs/about.md`](docs/about.md) |
| **Write-up** | [How good is speech-to-text at Hebrew words in English? I built a benchmark to find out](https://blog.danielrosehill.com/posts/english-hebrew-code-switching-asr-benchmark) (blog, 2026-10-15) |

## Results (test set v2, clean subset: 129 clips, 221 Hebrew term occurrences)

<!-- results:start -->
| Model | Type | Term recall % | Unseen-term recall % | Lenient % | WER % | English control WER % |
|---|---|---|---|---|---|---|
| [Parakeet v2 EN-HE Sabich v2.1](https://huggingface.co/Hebrish/Sabich) | CS fine-tune | **65.6** | 56.2 | 67.4 | 5.2 | 2.4 |
| [Parakeet v2 EN-HE Sabich v2.0](https://huggingface.co/Hebrish/Sabich) | CS fine-tune | **65.2** | 54.0 | 66.5 | 5.6 | 2.1 |
| [Parakeet v2 EN-HE Sabich v1.0](https://huggingface.co/Hebrish/Sabich) | CS fine-tune | **63.3** | 53.3 | 65.2 | 5.7 | 2.1 |
| [Parakeet v2 EN-HE Jachnun v1.0](https://huggingface.co/danielrosehill/Parakeet-TDT-0.6B-v2-English-Hebrew-Codeswitching) | CS fine-tune | **61.1** | 51.8 | 62.9 | 5.9 | 2.4 |
| [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Stock | **19.9** | 16.1 | 23.5 | 13.7 | 2.7 |
<!-- results:end -->

![Hebrew term recall by model, clean subset](docs/figures/term_recall.png)

**Term recall** is the share of Hebrew words written with the canonical romanized spelling, as a whole word and case-insensitive. It is the headline number, because a model can score a low WER and still mangle every Hebrew word. **Unseen-term recall** counts only words absent from the public training split. **Lenient** recall also accepts lexicon spelling variants. ⚠ marks a model with known training-data overlap: see the leaderboard. All test clips come from one speaker reading aloud, so treat gaps of a few points as noise ([limits](docs/about.md#limits)).

## Run it

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[whisper]'          # or '.[nemo]' for Parakeet / Canary
ehcs-bench run --model whisper:openai/whisper-large-v3-turbo --name "Whisper Large V3 Turbo"
ehcs-bench run --model nemo:nvidia/parakeet-tdt-0.6b-v2 --name "Parakeet TDT 0.6B v2"
```

A model spec is `<adapter>:<hf repo>[@revision]`; the resolved revision is recorded in the result. On a Modal GPU, with the NeMo build known to work:

```bash
MODAL_PROFILE=danielrosehill modal run modal_app.py \
  --model nemo:danielrosehill/Parakeet-TDT-0.6B-v2-English-Hebrew-Codeswitching \
  --name "Parakeet v2 EN-HE code-switching (v2)" --args "--fine-tuned --base-model nvidia/parakeet-tdt-0.6b-v2"
```

**Models that do not run here** (APIs, phone apps, GGUF runtimes): transcribe every WAV of the pinned test set yourself (v2: the 179 in `data/v2/audio/`) and write one `{"id": ..., "hyp": ...}` per line. Then run:

```bash
ehcs-bench score --preds preds.jsonl --name "My model" --model-id org/model
```

Every command takes `--version v1|v2` (default v2, the current set). v2 results are written to `results/v2/<name>.json`; v1 results stay in `results/`. `ehcs-bench rescore results/*.json results/v2/*.json` recomputes every metric, each file against the test set it records, from the stored transcripts and prints `OK` when it matches.

## Adding a result to the leaderboard

Put the result file in `results/`. Then run `python3 scripts/results_table.py` to refresh the table above, `scripts/make_figures.py` to redraw the charts, and `scripts/deploy_space.sh` to publish the Space. When a batch of new results is complete, freeze it as a dated round: `python3 scripts/snapshot_run.py --title "Round N: ..." --notes "..."`, commit, then deploy. Outside contributors open a PR with the result file. See [submitting](docs/about.md#submitting-a-result).

## Layout

| Path | What |
|---|---|
| `ehcs_bench/score.py` | Normaliser, term matcher, all metrics. CPU only. |
| `ehcs_bench/adapters.py` | `nemo` and `whisper` model adapters. Add new families here. |
| `ehcs_bench/data.py` | Test-set versions (v1, v2), their pinned commits, and the loader |
| `ehcs_bench/cli.py` | `ehcs-bench run / score / rescore` |
| `modal_app.py` | GPU runner on Modal |
| `results/` | One JSON per model: model metadata, metrics, every transcript. The latest result per model. v1 results at the top level, v2 results in `results/v2/` |
| `archive/` | One JSON per published benchmark round (`scripts/snapshot_run.py`). Append-only; the Space's *Archive* tab |
| `docs/data-dictionary.md` | Every term, field and metric. The Space's *Data dictionary* tab |
| `scripts/make_figures.py` | Static PNG charts into `docs/figures/` (`uv run --with matplotlib python scripts/make_figures.py`) |
| `scripts/build_testset.py` | Rebuilds test set v1 from pinned upstream commits |
| `scripts/build_testset_v2.py` | Rebuilds test set v2 (domain-balanced) from pinned upstream commits and `scripts/inputs/` (frozen lexicon, trained-on exclusions) |
| `docs/dataset-card/README.md` | Source of the HF dataset card. Upload it with `hf upload ... README.md --repo-type dataset` (a card-only commit; the pinned revision does not change) |
| `scripts/deploy_space.sh` | Publishes `space/` + `results/` (every test set version, current one as the default tab) to the leaderboard Space |
| `space/app.py` | The Gradio leaderboard |

## Related

- [Parakeet code-switching fine-tune tracker](https://github.com/Hebrish-ASR/English-Hebrew-Codeswitching-Parakeet-Finetune)
- [Whisper-Hebrish](https://huggingface.co/danielrosehill/Whisper-Hebrish)
- [English-Hebrew Code-Switching Index](https://github.com/danielrosehill/English-Hebrew-Code-Switching-Index)

Code MIT. Test-set audio CC-BY-4.0.
