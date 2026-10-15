# CLAUDE.md

Benchmark runner/scorer for English-Hebrew code-switching ASR. Created 2026-10-15. Public.
Read `docs/about.md` first: it is the methodology and also the Space's About tab.

## Where each artefact lives

| Artefact | Where | Changes how |
|---|---|---|
| Test set | HF dataset `danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark` | Never in place. `ehcs_bench/data.py` pins `REVISION`; a new test set is a new `VERSION` + `REVISION`, and old results stop showing on the Space (it filters by revision) |
| Results | `results/*.json` in **this** repo (source of truth) | Add a file; `scripts/results_table.py`; `scripts/deploy_space.sh` |
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

## Modal

Personal workspace only: `MODAL_PROFILE=danielrosehill`. Uses the existing secret
`parakeet-finetune-hf`. No Volumes, by Daniel's choice: weights download on every run. The image
copies the NeMo 2.7.3 pins from `~/repos/github/audio-dictation/Parakeet-V2-Personal-Finetune/training/app.py`.

## Upstream fixes owed

The 4 non-canonical references (`heder madregot`, `machane Yehuda`) are corrected here at
build time. They are still wrong in `Read-Aloud-Voice-Sentences` and should be fixed there
by its normaliser (private repo Parakeet-V2-Personal-Finetune, `scripts/normalize_transcripts.py`).

## Article drafts

`drafts/` is gitignored: the HF community article draft lives there (`drafts/hf-article.md`,
figures via the `drafts/figures -> ../docs/figures` symlink). HF has no API or CLI for creating
community blog articles or Posts (checked 2026-10-15: `huggingface_hub` 1.25.1, `hf` CLI, and
`/.well-known/openapi.json` only expose delete/comment endpoints for them), so publishing is a
manual paste into huggingface.co/new-blog.
