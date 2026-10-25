## What is measured

English speakers in Israel drop Hebrew words into English: *"grab some bamba from the makolet"*, *"I have to go to Bituach Leumi"*. English ASR has rarely heard these words and writes the nearest English-sounding thing (*Macaulay* for *makolet*). This benchmark measures how often a model gets them right, and whether it still transcribes plain English well.

The references write each Hebrew word in **one canonical romanized spelling** in Latin letters (*makolet*, *teudat zehut*, *Yerushalayim*). Hebrew script in the output is never a hit.

## Test set v2 (current, 2026-10-25)

v2 is the set Hebrish uses for its own internal benchmarks, and the one new results are scored on. It is in `data/v2/` of [`Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark`](https://huggingface.co/datasets/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark) (config `v2`), pinned at dataset commit `48ec85a301dc`, and built by `scripts/build_testset_v2.py`. Its source is the held-out test split of `Read-Aloud-Voice-Sentences` at commit `2cec2ebc85e9` (the revision Sabich v2.1 trained on). All 179 clips are phone recordings from 2026-10-14 to 2026-10-25.

| Subset | Clips | Headline? | What it is |
|---|---|---|---|
| `clean` | 129 (14.8 min) | **Yes** | Code-switching sentences, about 10 per Hebrew vocabulary domain, never trained on by any Hebrish model |
| `english_control` | 25 (2.0 min) | Regression check | The same speaker, everyday English with no Hebrew: email, general sentences, homophones |
| `tech` | 25 (2.3 min) | No | English tech and AI dictation with no Hebrew. WER only |

**Why a v2.** v1 drew its headline from whatever the test split held on 2026-10-15, so a few kinds of vocabulary dominated it. Its term labels also included English tech and product words (see *Limits*). v2 samples the same number of clips from each domain of the [hebrish.app lexicon](https://hebrish.app/lexicon), so the headline weighs places, food, bureaucracy and the army equally. It labels only Hebrew lexicon entries. It also excludes every clip that any Hebrish model trained on.

**Clean clips per domain:**

| Domain | Clips | Term occurrences in these clips | of which unseen | Eligible clips |
|---|---|---|---|---|
| `places` | 10 | 19 | 12 | 44 |
| `people` | 10 | 12 | 5 | 11 |
| `politics` | **9** | 15 | 12 | 9 |
| `religious` | 10 | 14 | 11 | 111 |
| `food` | 10 | 18 | 10 | 39 |
| `health` | 10 | 19 | 12 | 39 |
| `bureaucracy` | 10 | 21 | 12 | 23 |
| `money` | 10 | 16 | 9 | 23 |
| `home` | 10 | 18 | 12 | 37 |
| `army` | 10 | 15 | 11 | 20 |
| `transport` | 10 | 22 | 10 | 15 |
| `brands` | 10 | 15 | 11 | 19 |
| `everyday` | 10 | 17 | 10 | 43 |
| **Total** | **129** | **221** (187 distinct) | **137** | |

Politics had 9 eligible clips, one short of the target of 10. The other domains had enough.

**How it was built.** These are the steps in `scripts/build_testset_v2.py`. Its counts are in `data/v2/build-report.json` in the dataset.

1. **Eligible takes.** Of the 838 test-split rows, a take is eligible if it is the owner's own (`speaker` Daniel Rosehill or missing), has a transcript, and is not a 2025 desktop-mic import (30 dropped). It must not be a re-reading of a sentence Whisper-Hebrish trained on (16 dropped). It must not be a special-delivery task take: spelling alphabet, singing, whispering, odd voices, repetitions, disfluency, numbers, long-form (49 dropped). Only one take per sentence is kept (13 duplicates dropped).
2. **Never trained on (checked 2026-10-25).** A take is dropped if its id or its normalised text (lowercase, hyphens to spaces, punctuation removed) is in `runs/<run>/train.jsonl` or `val.jsonl` on the Modal volume `parakeet-cs-finetune`, for any of the 22 runs there. Those runs include `r10-all` (Sabich v2.1), `r9-all` (Sabich v2.0), `r7-all` (Sabich v1.0) and `r6-en30` (Jachnun v1.0). 13 test-split takes were dropped. All 13 matched by text (the same sentence recorded again, in train), none by id. Each of r10-all, r9-all and r7-all contained all 13; r6-en30 contained 10. The manifests are private, so the per-run matches are committed in `scripts/inputs/v2-trained-exclusions.json`, and the builder rebuilds the same set from that file.
3. **Headline candidates.** These are code-switching takes (`code_switch` category) with no tech category, because code-switching and tech are kept in separate sentences (20 dropped). Each needs at least one labelled Hebrew term (59 dropped).
4. **Term labels.** `hebrew_terms` holds the terms from the take's source label (`target_terms`) that are entries in hebrish.app's lexicon. The lexicon is frozen in `scripts/inputs/hebrish-lexicon-2026-10-20.json` (3,642 terms in 13 domains, built 2026-10-20). Each term is written in the lexicon's canonical spelling, matched by word boundary, and carries its domain. English tech and product words are not in the lexicon, so v1's *JSON*, *grep*, *Proxmox* and *Tuya* problem cannot recur. Across the headline candidates, 49 distinct source labels (61 occurrences) are not lexicon entries and are left unlabelled. Most are Hebrew words the lexicon lacks (*Maccabi Zahav* in 5 takes, *Makor Baruch*, *moshav*, *yeshiva*). Some are spellings the lexicon has in another form (*abba* vs *aba*, *Modiin*, *Rishon LeTsion*). A few are not Hebrew (*Daniel Rosehill*, *8200*). The full list is `target_terms_not_labelled` in the build report. These words still count in WER. Lexicon terms the source label missed are not added, because several lexicon brands are also English words (*Yes*, *Max*, *Partner*) and a text scan would mislabel them. Spelling variants come from Hebrew-Latin-Token-Classifier at commit `0459b004a775`. `unseen` means the term is absent from the source train split's `target_terms` at `2cec2ebc85e9`.
5. **One domain per clip.** A clip's domain is the most common domain among its terms. `people` terms count only when the clip has nothing else, because a name is usually the addressee (*"Hi Gal, ..."*), not the subject. Ties go to the domain with fewer eligible clips (to help the scarce ones), then alphabetically. 63 of the 129 clips have terms from more than one domain. Domains are assigned per term by the lexicon, not by reading the sentence, so a few look odd: *Shas* (the Talmud, in context) and *machshev* are under brands, and *cheder madregot* is under army. These are the lexicon's labels, kept as they are.
6. **Sampling.** In each domain, candidates are ordered by `sha256("ehcs-v2:" + id)`. A first pass takes clips that add a term not yet chosen in that domain, and the rest of the order fills up to 10. `english_control` is the first 25 takes in the same order whose categories are only `general`, `email`, `message`, `medical`, `target_word` or `homophone`, and which carry no Hebrew category or lexicon term. `tech` is the first 25 takes with a tech category and no Hebrew category. Rerunning the builder gave identical files.
7. **References and audio.** Two source transcripts spelled *heder madregot* where the label is the canonical *cheder madregot*. The reference uses the canonical term and keeps the original in `source_text`. Audio is 16 kHz mono 16-bit WAV. The 66 clips recorded as 48 kHz / 24-bit FLAC (from 2026-10-22) were downmixed and resampled with `scipy.signal.resample_poly`, as ASR-Training's `cs_lib.to_16k_wav` does for training.
8. **Oracle check.** Scoring the references as transcripts gives 0% WER, CER and raw WER on all three subsets. It gives 100% strict, lenient and exact-case term recall on all, unseen and seen terms, and in every domain. It was run on the copy downloaded from the Hub at `48ec85a301dc`.

v2 overlaps v1 in 24 `clean` clips and 5 English clips. The rest are new.

**Scoring.** A v2 result carries `metrics.subsets.clean`, `english_control` and `tech`. There is no `all_code_switch` in v2, because no contaminated clips are kept. A new `metrics.domains.<domain>` block gives `n`, `wer` and term recall per domain over `clean`. Each per-domain figure rests on about 10 clips and 12 to 22 term occurrences. Read it as a pointer to where a model is weak, not as a measurement.

## Test set v1 (2026-10-15, history)

v1 is kept unchanged in `data/test/` and pinned at dataset commit `3cb0b2b3b3e4`; every result scored on it stays valid and can be rescored. It was the leaderboard set until 2026-10-25. What follows is its original description.

145 clips, frozen at a pinned commit of [`Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark`](https://huggingface.co/datasets/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark). They are the held-out test split of `Read-Aloud-Voice-Sentences` (commit `3bdaa2a50d52`). Each clip is tagged with a subset:

| Subset | Clips | Headline? | Why |
|---|---|---|---|
| `clean` | 84 (11.8 min) | **Yes** | Phone recordings (2026-10) of sentences that appear in no published training set |
| `contaminated_text` | 7 | No | New recordings of sentences that Whisper-Hebrish trained on |
| `contaminated_audio` | 26 | No | Recordings that are *in* Whisper-Hebrish's training set (the identical files) |
| `desktop_mic` | 4 | No | Clean 2025 desktop-mic recordings, too few to be a subset of their own |
| `english_control` | 24 (3.3 min) | Regression check | The same speaker, English only: email, tech, names, homophones |

**How contamination was established (2026-10-15).** Whisper-Hebrish was trained on 2025-11-17 on `English-Hebrew-Mixed-Sentences` at commit `e7dc91ef7301`, using its `whisper_train` and `whisper_validation` files. 26 of the 30 imported desktop clips in the test split have ids in those files. 7 of the phone clips re-read a sentence from those files (normalised text match). Later commits of that dataset (2026-10-14) appended phone takes, so comparing against its `main` branch gives the wrong answer. The Parakeet fine-tune's test split is term-disjoint by design: no test clip is in its training data.

**Term labels.** Every clip lists its Hebrew terms with the accepted spelling variants from the [Hebrew-Latin-Token-Classifier lexicon](https://github.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier) and an `unseen` flag. The flag is set when the term is absent from the source dataset's *train* split. The clean subset has 172 term occurrences (139 distinct). 117 of the occurrences are unseen. `unseen` is defined against that one public train split, so it fits models trained on it, such as the Parakeet fine-tune. Whisper-Hebrish was trained on different data: 19 clean occurrences are of terms that appear in its training sentences, and 10 of those are flagged unseen.

Four source transcripts had a variant spelling where their own term label was canonical (*heder madregot*, *machane Yehuda*). With them, even a perfect transcript could not score a hit. The benchmark references use the canonical term and keep the original in `source_text`. An oracle check (references scored as transcripts) gives 0% WER and 100% on every term metric. That check is the guard against this class of bug.

## Metrics

All on the `clean` subset unless stated. They are the same in v1 and v2.

| Metric | Definition |
|---|---|
| **Term recall** | Share of Hebrew term occurrences whose canonical spelling appears in the transcript as a whole word, case-insensitive, with hyphen and space interchangeable. Never substring: *chool* is not found in *school*. |
| **Unseen-term recall** | The same, restricted to terms absent from the public training split. This is the generalisation number. |
| Lenient recall | Also accepts the lexicon's variant spellings (*tsaharon* for *tzaharon*). It is fairer to stock models, which were never taught the canonical spelling. |
| Exact-case recall | A strict hit where the casing also matches the reference (*Bituach Leumi*). |
| WER / CER | Lowercased, hyphens to spaces, punctuation removed except in-word apostrophes (`jiwer` 3.1.0). `wer_raw` (cased, punctuated) is in the result files. |
| English control WER | WER on `english_control`. A fine-tune that damages English shows up here. |

The normalisation and term matcher are the same ones behind the Parakeet fine-tune's model card. That card's numbers are not directly comparable, though: it scored all 121 code-switching clips (the *All code-switching* tab) and counted only its own 65 held-out terms as unseen.

## Limits

- **One speaker.** The speaker is Irish-born, English-speaking and lives in Jerusalem. Results rank models on *this* speaker's code-switching. They do not show performance on other accents.
- **Read speech**, mostly on one phone microphone.
- **Small.** v2 has 129 headline clips and 221 term occurrences; v1 had 84 and 172. A difference of 2 to 3 percentage points in term recall is about 5 hits, which is within noise. Per-domain numbers (v2) rest on about 10 clips each.
- **v1 only: not every labelled term is Hebrew (found 2026-10-15, fixed in v2).** 22 of the 139 clean-subset terms are English or product names: tech jargon (*JSON*, *YAML*, *grep*, *venv*, *Proxmox*, *Ubuntu*, *sherpa-onnx*, *VS Code*, ...) and brands (*Tuya*, *Astro*, *Pinecone*). They are 26 of the 172 occurrences. They came from the source dataset's term labels, which tag hard vocabulary in general. Excluding them changes no ranks but lowers every model's term recall by 2 to 6 points (Parakeet v2 EN-HE (v2): 65.7% to 63.7%; Whisper Large V3: 45.9% to 39.7%). v2 labels only hebrish.app lexicon entries, so it does not have this problem.
- The canonical spelling is a convention. *Lenient recall* is there because a model writing *tsaharon* is not wrong in any absolute sense.

## Submitting a result

1. **Run the model.** NeMo and Hugging Face Whisper checkpoints run directly:
   `ehcs-bench run --model nemo:<repo>` or `--model whisper:<repo>`. It runs on a GPU with `modal run modal_app.py --model ...`.
2. **Anything else** (an API, a phone app, a GGUF runtime): transcribe every WAV of the pinned test set yourself (v2: the 179 files in `data/v2/audio/`). Write `{"id", "hyp"}` JSONL and run `ehcs-bench score --preds preds.jsonl --name ... --model-id ...`.
   `--version v1` scores against the old set instead; the default is the current set, v2. A v2 result file is written to `results/v2/<name>.json` (v1 results stay in `results/`).
3. Open a pull request on [GitHub](https://github.com/Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark) that adds the result file to `results/`. CI-free check: `ehcs-bench rescore results/<file>.json` must print `OK`.

Declare any overlap between your training data and this test set with `--contamination`. Training on the **train** split of the source datasets is fine and expected. That is what the `unseen` metric exists for.
