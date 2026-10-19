## Data dictionary

Every term, field and metric used by the benchmark, in one place. The methodology and the reasoning behind these choices are in *About*. The code that computes every metric is [`ehcs_bench/score.py`](https://github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark/blob/main/ehcs_bench/score.py).

### Concepts

| Term | Meaning |
|---|---|
| **Code-switching** | Using words from a second language inside a sentence in the first. Here: English sentences with Hebrew words in them (*"pick up some bamba from the makolet"*). |
| **Hebrew term** | A Hebrew word or multi-word phrase inside an English sentence (*makolet*, *Bituach Leumi*, *teudat zehut*). Proper nouns (people, places) count. |
| **Term occurrence** | One appearance of a term in one clip. Recall is computed over occurrences, so a term that appears 3 times counts 3 times. |
| **Canonical spelling** | The one romanized (Latin-letter) spelling the reference uses for a term. A convention, not a truth: see *lenient*. |
| **Variant** | An alternative romanization accepted by lenient recall (*tsaharon* for *tzaharon*). From the [Hebrew-Latin-Token-Classifier lexicon](https://github.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier). |
| **Unseen term** | A term absent from the *train* split of the source dataset `Read-Aloud-Voice-Sentences`. Measures generalisation for models trained on that split. |
| **Seen term** | A term that does appear in that train split. |
| **Hit** | The term is found in the transcript as a whole word, case-insensitive, hyphen and space interchangeable. Never a substring match (*chool* is not in *school*). Hebrew script is never a hit. |
| **Contamination** | Overlap between a model's training data and the test set: identical recordings (`contaminated_audio`) or new recordings of training sentences (`contaminated_text`). Those clips are excluded from the headline. |
| **Oracle check** | Scoring the references themselves as if they were transcripts. Must give 0% WER and 100% on every term metric; anything else is a bug in the test set or scorer. |
| **Benchmark round** | A dated snapshot of the leaderboard (*Archive* tab). Rounds are immutable once published. |

### Subsets

| Subset | Clips | In headline | Description |
|---|---|---|---|
| `clean` | 84 | **Yes** | Phone recordings (2026-10) of sentences that appear in no published training set |
| `contaminated_text` | 7 | No | New recordings of sentences Whisper-Hebrish trained on |
| `contaminated_audio` | 26 | No | Recordings that are in Whisper-Hebrish's training set |
| `desktop_mic` | 4 | No | 2025 desktop-microphone recordings |
| `english_control` | 24 | Regression check | Same speaker, English only, no Hebrew terms |

Scoring groups (keys under `metrics.subsets` in a result file):

| Group | Pools | Use |
|---|---|---|
| `clean` | `clean` | Headline leaderboard |
| `all_code_switch` | `clean` + `contaminated_text` + `contaminated_audio` + `desktop_mic` (121 clips) | Comparable with the Parakeet fine-tune's model card. Unfair to Whisper-Hebrish |
| `english_control` | `english_control` | Does a fine-tune damage plain English? |

### Metrics

| Metric | Field in result file | Definition | Better |
|---|---|---|---|
| **Term recall** | `terms.all.recall` | Hits ÷ term occurrences, canonical spelling | Higher |
| **Unseen-term recall** | `terms.unseen.recall` | Term recall over unseen terms only | Higher |
| Seen-term recall | `terms.seen.recall` | Term recall over seen terms only | Higher |
| **Lenient recall** | `terms.*.recall_lenient` | A hit on the canonical spelling *or* any lexicon variant | Higher |
| **Exact-case recall** | `terms.*.recall_exact_case` | A canonical hit whose casing also matches the reference (*Bituach Leumi*, not *bituach leumi*) | Higher |
| **WER** | `wer` | Word error rate after normalisation (lowercase, hyphens to spaces, punctuation removed except in-word apostrophes). `jiwer` 3.1.0 | Lower |
| **CER** | `cer` | Character error rate, same normalisation | Lower |
| Raw WER | `wer_raw` | WER on the cased, punctuated text, no normalisation | Lower |
| **English control WER** | `english_control.wer` | WER on the English-only clips | Lower |
| Occurrences | `terms.*.occurrences` | Number of term occurrences in the group (clean: 172) | — |
| Terms | `terms.*.terms` | Number of distinct terms (clean: 139) | — |

### Test-set fields (`metadata.jsonl` in the dataset)

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Clip id: UTC recording time plus a random suffix |
| `file_name` | string | Path of the 16 kHz mono WAV inside the dataset |
| `text` | string | Reference transcript, with canonical term spellings |
| `subset` | string | One of the subsets above |
| `duration_s` | float | Clip length in seconds |
| `hebrew_terms` | list | One entry per term occurrence: `term` (canonical), `variants` (list), `unseen` (bool) |
| `categories` | list | Content tags: `code_switch`, `tech`, `email`, `message`, `names`, `general`, `long`, `homophone` |
| `recording` | string | `phone_2026` or `desktop_mic_2025` |
| `source_dataset` | string | Upstream dataset and pinned commit the clip came from |
| `source_text` | string or null | The upstream transcript when the reference corrected its spelling, else null |

### Result-file fields (`results/<model>.json`)

| Field | Meaning |
|---|---|
| `schema` | Result format version (currently 1) |
| `benchmark.version`, `benchmark.revision` | Test-set version and the exact dataset commit it was scored against. The leaderboard only shows results for the current revision |
| `model.id`, `model.revision` | Model repo and the resolved commit that was run |
| `model.name` | Display name |
| `model.adapter` | `nemo`, `whisper`, or `external` (transcribed outside the runner) |
| `model.base_model` | What it was fine-tuned from, if anything |
| `model.fine_tuned_for_code_switching` | True for English-Hebrew code-switching fine-tunes |
| `model.params_m` | Parameters, millions |
| `model.settings` | Decoding settings (language, greedy/beam, chunk length, dtype) |
| `model.contamination` | Declared overlap between the model's training data and the test set |
| `scored_at` | UTC time the result was scored |
| `run.host`, `run.wall_seconds`, `run.audio_seconds` | Where it ran and how long it took for how much audio |
| `metrics.subsets.<group>` | `n`, `minutes`, `wer`, `cer`, `wer_raw`, and `terms` as above |
| `metrics.per_term.<term>` | Clean subset only: `n`, `strict`, `lenient`, `exact` hit counts and `unseen` |
| `metrics.missing` | Clip ids with no transcript (scored as empty) |
| `utterances[]` | Every clip: `id`, `subset`, `ref`, `hyp`. `ehcs-bench rescore` recomputes all metrics from these |

### Archive fields (`archive/<date>.json`)

| Field | Meaning |
|---|---|
| `date`, `title`, `notes` | When the round was published and what it contained |
| `benchmark.version`, `benchmark.revision` | Test set the round was scored on |
| `repo_commit` | GitHub commit holding the full result files (with transcripts) for the round |
| `models[]` | Per model: identity, revision, `contamination`, `scored_at`, `host`, and the `subsets` metrics block |
