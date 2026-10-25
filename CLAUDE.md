# CLAUDE.md

Benchmark runner/scorer for English-Hebrew code-switching ASR. Created 2026-10-15. Public.
Read `docs/about.md` first: it is the methodology and also the Space's About tab.

## Where each artefact lives

| Artefact | Where | Changes how |
|---|---|---|
| Test set | HF dataset `Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark` (public, ungated, CC-BY-4.0) | Never in place. Each version has its own folder and config (`v1`: `data/test/` at `3cb0b2b3b3e4`; `v2`: `data/v2/` at `48ec85a301dc`, current since 2026-10-25). `ehcs_bench/data.py` holds `VERSIONS` and the current `VERSION` + `REVISION`; a new test set is a new entry plus a new `VERSION` + `REVISION`, and old results stop showing on the Space (it filters by revision) |
| Results | `results/*.json` (v1) and `results/v2/*.json` (v2) in **this** repo (source of truth) | Add a file; `scripts/results_table.py`; `scripts/deploy_space.sh` |
| Archive | `archive/<date>.json`, one per published round | `scripts/snapshot_run.py` after the round's results are committed. Never edit a committed round |
| Leaderboard | HF Space `danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard` | Only via `scripts/deploy_space.sh`, which copies space/, results/, archive/, docs/about.md, docs/data-dictionary.md and the pinned metadata. Never edit the Space directly |

README edits to the HF dataset create new commits; that is fine because the loader pins
`REVISION`, not `main`. Do not repin unless the audio or metadata changed.

## Invariants

- The oracle check: scoring the references as transcripts must give 0 WER and 1.0 on every
  term metric. Run it after any change to `score.py` or the test set (make preds from
  `metadata.jsonl` text). It caught 4 references with a non-canonical spelling and 7 label/
  reference case mismatches on 2026-10-15.
- Term matching is word-boundary, never substring (see the 114 substring-matched labels in
  English-Hebrew-Mixed-Sentences).
- Contamination is computed against `English-Hebrew-Mixed-Sentences@e7dc91ef7301` (what
  Whisper-Hebrish trained on 2025-11-17), not `main`: `main` gained phone takes on 2026-10-14.

## Test set v2 (2026-10-25): state and what is still owed

- Built by `scripts/build_testset_v2.py` from `Read-Aloud-Voice-Sentences@2cec2ebc85e9` (gated: run with
  `with-secrets`). Inputs frozen in `scripts/inputs/`: the hebrish.app lexicon (its repo,
  `danielrosehill/Hebrish-Website`, is private, so it is copied here) and `v2-trained-exclusions.json`
  (test ids found in any Modal `parakeet-cs-finetune` `runs/<run>/{train,val}.jsonl`, 22 runs on 2026-10-25).
  Pass `--manifests DIR` only to recompute that file; v2 itself is frozen, so do not.
- **No model is scored on v2 yet** (scoring waits for the Sabich v2.1 release). Until then:
  `scripts/results_table.py` would empty the README table (it filters to the current revision), and
  `scripts/deploy_space.sh` stops at the `cp results/v2/*.json` step.
- **`space/app.py` is still v1-only and needs a pass before the first v2 deploy:** it builds an
  `all_code_switch` tab (v2 results have no such group, so it would raise KeyError), its subset dropdown lists
  v1 subsets, and its header, tab titles and chart captions hard-code v1 counts (84 clips, 172 occurrences,
  139 terms, the non-Hebrew-term notice). v2 adds `metrics.subsets.tech` and `metrics.domains` to show.
- The oracle check for v2 (from the Hub copy at `48ec85a301dc`) passed on 2026-10-25.

## Modal

Personal workspace only: `MODAL_PROFILE=danielrosehill`. Uses the existing secret
`parakeet-finetune-hf`. No Volumes, by Daniel's choice: weights download on every run. The image
copies the NeMo 2.7.3 pins from `~/repos/github/audio-dictation/ASR-Training/training/app.py`.

## Upstream fixes owed

The 4 non-canonical references (`heder madregot`, `machane Yehuda`) are corrected here at
build time (v2 adds the `nachal Oz` -> `Nahal Oz` variant to the same table, and corrected 2 `heder madregot` references). They are still wrong in `Read-Aloud-Voice-Sentences` and should be fixed there
by its normaliser (private repo ASR-Training, `scripts/normalize_transcripts.py`).

## Article drafts

`drafts/` is gitignored: the HF community article draft lives there (`drafts/hf-article.md`,
figures via the `drafts/figures -> ../docs/figures` symlink). HF has no API or CLI for creating
community blog articles or Posts (checked 2026-10-15: `huggingface_hub` 1.25.1, `hf` CLI, and
`/.well-known/openapi.json` only expose delete/comment endpoints for them), so publishing is a
manual paste into huggingface.co/new-blog.
