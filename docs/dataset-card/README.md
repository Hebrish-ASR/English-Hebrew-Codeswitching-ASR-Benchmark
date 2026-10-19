---
license: cc-by-4.0
language:
- en
- he
task_categories:
- automatic-speech-recognition
pretty_name: English-Hebrew Code-Switching ASR Benchmark
tags:
- speech
- asr
- benchmark
- evaluation
- code-switching
- hebrew
- israel
- codeswitching
- speech-recognition
- hebrew-nlp
- english-hebrew
- english
- single-speaker
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test/**
---

# English-Hebrew Code-Switching ASR Benchmark (test set v1)

> **Status: early-stage, work in progress.** Known issue: 22 of the 139 clean-subset `hebrew_terms` labels are English tech words or product names (*JSON*, *grep*, *Proxmox*, *Tuya*, ...), 26 of 172 occurrences. v1.1 will relabel them; until then, term recall is 2 to 6 points higher than a Hebrew-only count. Details: [methodology, limits](https://github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark/blob/main/docs/about.md#limits).

A frozen, versioned test set for measuring how well speech recognition handles **English speech with Hebrew words in it**: *"pick up some bamba from the makolet"*, *"I have to go to Bituach Leumi tomorrow"*. The references write each Hebrew word in **one canonical romanized spelling**, in Latin letters.

- **Website (the Parakeet code-switching fine-tune):** [hebrish.app](https://hebrish.app): get the Hebrish keyboard for Android and the desktop apps, see the results, and find every model, dataset and demo in one place.
- **Runner and scorer:** [github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark](https://github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark)
- **Leaderboard:** [Space: English-Hebrew-Codeswitching-ASR-Leaderboard](https://huggingface.co/spaces/danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard)

This repository is the test set only. It never changes in place: a new test set is a new version, and every result records the commit it was scored against.

## Subsets

145 clips from the held-out **test** split of [`danielrosehill/Read-Aloud-Voice-Sentences`](https://huggingface.co/datasets/danielrosehill/Read-Aloud-Voice-Sentences) at commit `3bdaa2a50d52`, each tagged with a `subset`:

| `subset` | Clips | What it is | Used for |
|---|---|---|---|
| `clean` | 84 (11.8 min) | Phone recordings (2026-10-14/15) of sentences that appear in no published training set | **The leaderboard headline** |
| `contaminated_text` | 7 | New phone recordings of sentences that Whisper-Hebrish was trained on (different audio, same words) | Reported only in `all_code_switch` |
| `contaminated_audio` | 26 | Desktop-mic recordings (2025-11-17) that are *in* Whisper-Hebrish's training set, the same files | Reported only in `all_code_switch` |
| `desktop_mic` | 4 | Desktop-mic recordings from the same 2025 round that no model trained on | Reported only in `all_code_switch` |
| `english_control` | 24 (3.3 min) | The same speaker's English with no Hebrew: email, tech, names, homophones | English regression check |

The `clean` subset holds 172 Hebrew term occurrences (139 distinct terms). 117 of those occurrences are of terms that appear nowhere in the training split of the source dataset (`unseen: true`), so they measure generalisation to new Hebrew words rather than memorisation.

## Columns

| Column | Meaning |
|---|---|
| `file_name` | WAV, 16 kHz mono |
| `id` | Same id as in the source dataset |
| `text` | Reference transcript: cased and punctuated, canonical romanized Hebrew |
| `subset` | See above |
| `hebrew_terms` | List of `{term, variants, unseen}`. `term` is the canonical spelling. `variants` are accepted alternative romanizations from the [Hebrew-Latin-Token-Classifier lexicon](https://github.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier). `unseen` is true when the term is absent from the source dataset's train split |
| `source_text` | Set on the 4 rows whose reference was corrected: the source transcript had a variant spelling (*heder madregot*, *machane Yehuda*) where its own label was the canonical term, so no transcript could have scored a hit. The reference here uses the canonical term; `source_text` is the original |
| `duration_s`, `categories`, `recording`, `source_dataset` | Provenance |

## Speaker and limits

One speaker: Daniel Rosehill, the author, Irish-born, English-speaking, living in Jerusalem. Phone clips were recorded on a OnePlus Nord 3 5G internal microphone. This is a **single-speaker, read-speech** benchmark of 84 headline clips. It ranks models on this speaker's code-switching; it does not show how a model handles other accents or spontaneous speech. Differences of a few term hits are within noise.

## Licence

CC-BY-4.0, like the source dataset. Built by `scripts/build_testset.py` in the GitHub repository from pinned upstream commits.
