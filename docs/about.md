## What is measured

English speakers in Israel drop Hebrew words into English: *"grab some bamba from the makolet"*, *"I have to go to Bituach Leumi"*. English ASR has rarely heard these words and writes the nearest English-sounding thing (*Macaulay* for *makolet*). This benchmark measures how often a model gets them right, and whether it still transcribes plain English well.

The references write each Hebrew word in **one canonical romanized spelling** in Latin letters (*makolet*, *teudat zehut*, *Yerushalayim*). Hebrew script in the output is never a hit.

## Test set (v1)

145 clips, frozen at a pinned commit of [`danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark`](https://huggingface.co/datasets/danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark). They are the held-out test split of `Read-Aloud-Voice-Sentences` (commit `3bdaa2a50d52`). Each clip is tagged with a subset:

| Subset | Clips | Headline? | Why |
|---|---|---|---|
| `clean` | 84 (11.8 min) | **Yes** | Phone recordings (2026-10) of sentences that appear in no published training set |
| `contaminated_text` | 7 | No | New recordings of sentences that Whisper-Hebrish trained on |
| `contaminated_audio` | 26 | No | Recordings that are *in* Whisper-Hebrish's training set (the identical files) |
| `desktop_mic` | 4 | No | Clean 2025 desktop-mic recordings, too few to be a subset of their own |
| `english_control` | 24 (3.3 min) | Regression check | The same speaker, English only: email, tech, names, homophones |

**How contamination was established (2026-10-15).** Whisper-Hebrish was trained on 2025-11-17 on `English-Hebrew-Mixed-Sentences` at commit `e7dc91ef7301`, using its `whisper_train` and `whisper_validation` files. 26 of the 30 imported desktop clips in the test split have ids in those files. 7 of the phone clips re-read a sentence from those files (normalised text match). Later commits of that dataset (2026-10-14) appended phone takes, so comparing against its `main` branch gives the wrong answer. The Parakeet fine-tune's test split is term-disjoint by design: no test clip is in its training data.

**Term labels.** Every clip lists its Hebrew terms with the accepted spelling variants from the [Hebrew-Latin-Token-Classifier lexicon](https://github.com/danielrosehill/Hebrew-Latin-Token-Classifier) and an `unseen` flag. The flag is set when the term is absent from the source dataset's *train* split. The clean subset has 172 term occurrences (139 distinct). 117 of the occurrences are unseen. `unseen` is defined against that one public train split, so it fits models trained on it, such as the Parakeet fine-tune. Whisper-Hebrish was trained on different data: 19 clean occurrences are of terms that appear in its training sentences, and 10 of those are flagged unseen.

Four source transcripts had a variant spelling where their own term label was canonical (*heder madregot*, *machane Yehuda*). With them, even a perfect transcript could not score a hit. The benchmark references use the canonical term and keep the original in `source_text`. An oracle check (references scored as transcripts) gives 0% WER and 100% on every term metric. That check is the guard against this class of bug.

## Metrics

All on the `clean` subset unless stated.

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
- **Small.** 84 headline clips and 172 term occurrences. A difference of 2 to 3 percentage points in term recall is about 4 to 5 hits, which is within noise.
- **Not every labelled term is Hebrew (found 2026-10-15, open).** 22 of the 139 clean-subset terms are English or product names: tech jargon (*JSON*, *YAML*, *grep*, *venv*, *Proxmox*, *Ubuntu*, *sherpa-onnx*, *VS Code*, ...) and brands (*Tuya*, *Astro*, *Pinecone*). They are 26 of the 172 occurrences. They came from the source dataset's term labels, which tag hard vocabulary in general. Excluding them changes no ranks but lowers every model's term recall by 2 to 6 points (Parakeet v2 EN-HE (v2): 65.7% to 63.7%; Whisper Large V3: 45.9% to 39.7%). A relabelled test set will be a new version.
- The canonical spelling is a convention. *Lenient recall* is there because a model writing *tsaharon* is not wrong in any absolute sense.

## Submitting a result

1. **Run the model.** NeMo and Hugging Face Whisper checkpoints run directly:
   `ehcs-bench run --model nemo:<repo>` or `--model whisper:<repo>`. It runs on a GPU with `modal run modal_app.py --model ...`.
2. **Anything else** (an API, a phone app, a GGUF runtime): transcribe the 145 WAVs yourself. Write `{"id", "hyp"}` JSONL and run `ehcs-bench score --preds preds.jsonl --name ... --model-id ...`.
3. Open a pull request on [GitHub](https://github.com/danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark) that adds the result file to `results/`. CI-free check: `ehcs-bench rescore results/<file>.json` must print `OK`.

Declare any overlap between your training data and this test set with `--contamination`. Training on the **train** split of the source datasets is fine and expected. That is what the `unseen` metric exists for.
