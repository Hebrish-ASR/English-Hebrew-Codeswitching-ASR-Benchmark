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
- config_name: v2
  default: true
  data_files:
  - split: test
    path:
    - data/v2/metadata.jsonl
    - data/v2/audio/*.wav
- config_name: v1
  data_files:
  - split: test
    path: data/test/**
---

# English-Hebrew Code-Switching ASR Benchmark

> **Status (2026-10-25): test set v2 is current.** It is the set Hebrish uses for its own internal benchmarks, and the one new results are scored on. Still a small, single-speaker benchmark: read the limits below before drawing conclusions. v1 (2026-10-15) stays here unchanged, with the results scored on it.

A frozen, versioned test set for measuring how well speech recognition handles **English speech with Hebrew words in it**: *"pick up some bamba from the makolet"*, *"I have to go to Bituach Leumi tomorrow"*. The references write each Hebrew word in **one canonical romanized spelling**, in Latin letters.

- **Website (the Parakeet code-switching fine-tune):** [hebrish.app](https://hebrish.app): get the Hebrish keyboard for Android and the desktop apps, see the results, and find every model, dataset and demo in one place.
- **Runner and scorer:** [github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark](https://github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark)
- **Leaderboard:** [Space: English-Hebrew-Codeswitching-ASR-Leaderboard](https://huggingface.co/spaces/danielrosehill/English-Hebrew-Codeswitching-ASR-Leaderboard)

This repository is the test set only. It never changes in place: a new test set is a new version in its own folder, and every result records the version and the commit it was scored against.

| Version | Folder | Config | Pinned commit | Status |
|---|---|---|---|---|
| **v2** (2026-10-25) | `data/v2/` | `v2` (default) | `48ec85a301dc` | **Current.** Domain-balanced code-switching, 179 clips |
| v1 (2026-10-15) | `data/test/` | `v1` | `3cb0b2b3b3e4` | History. 145 clips; results scored on it stay valid against that commit |

```python
from datasets import load_dataset
v2 = load_dataset("Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark", "v2", split="test")
```

<!-- hebrish-listen:begin -->
## Listen: what Hebrish is and how it's made

A spoken introduction to the project by the person who makes it: why Hebrish exists, why stock
speech-to-text gets Hebrew words wrong, how the fine-tunes are trained, the model lines (Sabich,
Jachnun, Shakshuka, Meorav), how the training data is recorded, and the plans for the apps.

| | |
| --- | --- |
| Listen | [hebrish.app/updates](https://hebrish.app/updates): player, chapters, edited transcript, MP3 download |
| Author | Daniel Rosehill |
| Recorded | 2026-10-22 |
| Runtime | 1:07:03 |

Chapters (each opens the player at that point):
[0:00](https://hebrish.app/updates#t=0) Why this voice note ·
[1:36](https://hebrish.app/updates#t=96) Coming to speech to text ·
[6:36](https://hebrish.app/updates#t=396) Why Hebrew words go wrong ·
[9:23](https://hebrish.app/updates#t=563) How the fine-tune works ·
[12:27](https://hebrish.app/updates#t=747) Why Parakeet, and the model plan ·
[17:44](https://hebrish.app/updates#t=1064) Disfluencies, shekels, paragraphs ·
[25:40](https://hebrish.app/updates#t=1540) Tech vocabulary and Meorav ·
[29:07](https://hebrish.app/updates#t=1747) The kitchen sink: Sabich and Jachnun ·
[33:35](https://hebrish.app/updates#t=2015) Licensing, accents, noisy recordings ·
[39:10](https://hebrish.app/updates#t=2350) One recorder for ASR and TTS ·
[41:04](https://hebrish.app/updates#t=2464) Desktop shortcuts and the macro pad ·
[43:34](https://hebrish.app/updates#t=2614) Android: Bluetooth and one button ·
[45:35](https://hebrish.app/updates#t=2735) Streaming and Shakshuka ·
[50:28](https://hebrish.app/updates#t=3028) Long-form transcription ·
[56:00](https://hebrish.app/updates#t=3360) Output modes ·
[63:01](https://hebrish.app/updates#t=3781) Report a word, and more voices
<!-- hebrish-listen:end -->

## Test set v2 (current)

179 clips from the held-out **test** split of [`danielrosehill/Read-Aloud-Voice-Sentences`](https://huggingface.co/datasets/danielrosehill/Read-Aloud-Voice-Sentences) at commit `2cec2ebc85e9`, all phone recordings from 2026-10-14 to 2026-10-25, each tagged with a `subset`:

| `subset` | Clips | What it is | Used for |
|---|---|---|---|
| `clean` | 129 (14.8 min) | Code-switching sentences, about 10 per Hebrew vocabulary domain, none of them trained on by any Hebrish model | **The headline** |
| `english_control` | 25 (2.0 min) | The same speaker's everyday English with no Hebrew: email, general sentences, homophones | English regression check |
| `tech` | 25 (2.3 min) | English tech and AI dictation with no Hebrew (*npm*, *Proxmox*, *Claude Code*) | Tech vocabulary check, kept out of the headline |

### Clean clips per domain

The domains are the ones [hebrish.app's lexicon](https://hebrish.app/lexicon) uses. Each clip has one `domain`, assigned from its Hebrew terms (rule below).

| `domain` | Lexicon label | Clips | Minutes | Term occurrences in these clips | of which unseen | Clips available |
|---|---|---|---|---|---|---|
| `places` | Places & streets | 10 | 1.2 | 19 | 12 | 44 |
| `people` | People & public figures | 10 | 0.9 | 12 | 5 | 11 |
| `politics` | Politics & news | **9** | 1.1 | 15 | 12 | 9 |
| `religious` | Religious life & calendar | 10 | 1.3 | 14 | 11 | 111 |
| `food` | Food & shopping | 10 | 1.2 | 18 | 10 | 39 |
| `health` | Health & pharmacy | 10 | 0.9 | 19 | 12 | 39 |
| `bureaucracy` | Bureaucracy & government | 10 | 1.3 | 21 | 12 | 23 |
| `money` | Money & work | 10 | 1.0 | 16 | 9 | 23 |
| `home` | Home, family & school | 10 | 1.2 | 18 | 12 | 37 |
| `army` | Army & security | 10 | 1.2 | 15 | 11 | 20 |
| `transport` | Transport & travel | 10 | 1.3 | 22 | 10 | 15 |
| `brands` | Brands & acronyms | 10 | 1.2 | 15 | 11 | 19 |
| `everyday` | Everyday talk | 10 | 1.0 | 17 | 10 | 43 |
| **Total** | | **129** | **14.8** | **221** (187 distinct terms) | **137** | |

*Clips available* is the eligible pool after every filter below; politics had only 9, so it has 9.

**How the clean clips were chosen.**

- **Eligible:** the speaker's own phone takes with a transcript, category `code_switch`, no tech category (code-switching and tech are kept in separate sentences), no special-delivery task (spelling alphabet, singing, whispering, numbers, long-form), not a 2025 desktop-mic import, not a sentence Whisper-Hebrish trained on, and one take per sentence.
- **Never trained on:** a clip is out if its id or its normalised text is in the training or validation manifest of any Hebrish training run up to 2026-10-25 (22 runs, including Sabich v2.1, v2.0 and v1.0 and Jachnun v1.0). 13 test-split clips were excluded this way. All 13 matched by text (the same sentence re-recorded), none by id.
- **Hebrew terms:** a clip's `hebrew_terms` are the terms from its source label that are entries in hebrish.app's lexicon. English tech and product words (*JSON*, *grep*, *Proxmox*, *Tuya*: v1's labelling issue) are not lexicon entries, so they are not labelled. Hebrew words the lexicon lacks (*Maccabi Zahav*, *Makor Baruch*) are left unlabelled too. They stay in the reference text and count towards WER only.
- **Domain rule:** the most common domain among the clip's terms. `people` terms count only when the clip has no other kind, because a name is usually just the addressee (*"Hi Gal, ..."*). A tie goes to the domain with fewer eligible clips, then alphabetically. 63 of the 129 clips have terms from more than one domain. Each term in `hebrew_terms` also carries its own `domain`.
- **Sampling:** in each domain, clips are ordered by `sha256("ehcs-v2:" + id)`. The first pass takes clips that add a term not yet chosen in that domain, then the rest of the order fills up to 10. The English and tech companions are the first 25 of their pools in the same order. A rerun of the builder gives the same set.
- **Unseen:** a term is `unseen` when it is absent from the source dataset's train split at the same commit. 137 of the 221 occurrences are unseen.
- **References:** 2 source transcripts had a variant spelling (*heder madregot*) where the label is the canonical term. The reference uses the canonical term and keeps the original in `source_text`. An oracle check (references scored as transcripts) gives 0% WER and 100% on every term metric for every subset and domain.

## Columns (v2)

| Column | Meaning |
|---|---|
| `file_name` | `audio/<id>.wav`, 16 kHz mono 16-bit. Clips recorded as 48 kHz / 24-bit FLAC (from 2026-10-22) were resampled |
| `id` | Same id as in the source dataset |
| `text` | Reference transcript: cased and punctuated, canonical romanized Hebrew |
| `subset` | `clean`, `english_control` or `tech` |
| `domain` | Lexicon domain of a `clean` clip; null otherwise |
| `hebrew_terms` | List of `{term, variants, unseen, domain}`. `term` is the lexicon's canonical spelling. `variants` are accepted alternative romanizations from the [Hebrew-Latin-Token-Classifier lexicon](https://github.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier). `unseen` is true when the term is absent from the source train split. Empty for the English subsets |
| `source_text` | The original transcript when the reference corrected a spelling, else null |
| `source_audio` | Format of the source file (`wav 16000 Hz PCM_16`, `flac 48000 Hz PCM_24`) |
| `duration_s`, `categories`, `recording`, `source_dataset` | Provenance |

`data/v2/build-report.json` holds every filter count, the per-domain pools and the contamination check.

## Test set v1 (history)

145 clips from the test split of `Read-Aloud-Voice-Sentences` at commit `3bdaa2a50d52`, in `data/test/`, pinned at commit `3cb0b2b3b3e4` of this repository. Its `clean` headline was 84 clips with 172 term occurrences; the other subsets were `contaminated_text` (7), `contaminated_audio` (26), `desktop_mic` (4) and `english_control` (24). Its columns are those of v2 without `domain` and `source_audio`.

Known issue, fixed in v2: 22 of v1's 139 clean-subset terms were English tech words or product names (*JSON*, *grep*, *Proxmox*, *Tuya*, ...), 26 of 172 occurrences, which put term recall 2 to 6 points above a Hebrew-only count. v1 was not balanced by domain. Full v1 description: [docs/about.md](https://github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark/blob/main/docs/about.md) on GitHub.

## Speaker and limits

One speaker: Daniel Rosehill, the author, Irish-born, English-speaking, living in Jerusalem. Clips were recorded on a OnePlus Nord 3 5G internal microphone. This is a **single-speaker, read-speech** benchmark of 129 headline clips, about 10 per domain. It ranks models on this speaker's code-switching; it does not show how a model handles other accents or spontaneous speech. A per-domain score rests on 10 clips and 12 to 22 term occurrences, so treat it as a pointer, not a measurement. Overall, differences of a few term hits are within noise. Domains come from the lexicon by term, not by sentence context, so a few assignments look odd (*Shas* is filed under brands, *machshev* too).

## Licence

CC-BY-4.0, like the source dataset. v2 is built by `scripts/build_testset_v2.py`, v1 by `scripts/build_testset.py`, in the GitHub repository from pinned upstream commits.
