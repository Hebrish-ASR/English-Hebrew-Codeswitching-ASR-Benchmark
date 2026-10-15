"""Model adapters. Each takes a list of 16 kHz mono WAV paths and returns one transcript per
path, in order. Add a new family here as a function and register it in ADAPTERS.

A model spec on the command line is `<adapter>:<hf repo id>[@revision]`, for example
`nemo:nvidia/parakeet-tdt-0.6b-v2` or `whisper:openai/whisper-large-v3-turbo@<sha>`.
Anything that cannot run here (an API, a phone app, a GGUF runtime) is benchmarked by
writing a predictions file and using `ehcs-bench score` instead.
"""


def parse_spec(spec):
    adapter, _, rest = spec.partition(":")
    if adapter not in ADAPTERS or not rest:
        raise SystemExit(f"model spec must be <adapter>:<repo>[@rev], adapter one of {sorted(ADAPTERS)}; got {spec!r}")
    repo, _, rev = rest.partition("@")
    return adapter, repo, rev or None


def resolve_revision(repo, revision):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, revision=revision).sha


def nemo(repo, revision, paths, batch_size=16, **_):
    """NVIDIA NeMo models (Parakeet, Canary, Nemotron). Loads the single .nemo file in the
    repo at the pinned revision, so fine-tunes published as a bare .nemo work the same as
    NVIDIA's own repos."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    import nemo.collections.asr as nemo_asr
    files = [f.rfilename for f in HfApi().model_info(repo, revision=revision).siblings if f.rfilename.endswith(".nemo")]
    if len(files) != 1:
        raise SystemExit(f"{repo}: expected exactly one .nemo file, found {files}")
    m = nemo_asr.models.ASRModel.restore_from(hf_hub_download(repo, files[0], revision=revision))
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda()
    out = m.transcribe(paths, batch_size=batch_size, verbose=False)
    if isinstance(out, tuple):
        out = out[0]
    return [o.text if hasattr(o, "text") else str(o) for o in out], {"nemo_file": files[0], "decoding": "default (greedy)"}


def whisper(repo, revision, paths, batch_size=16, language="en", **_):
    """Hugging Face transformers Whisper checkpoints. Language is forced to English: the
    references are English with romanized Hebrew, and auto-detect sometimes switches a whole
    sentence to Hebrew script."""
    import torch
    from transformers import pipeline
    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=repo, revision=revision,
                   torch_dtype=torch.float16 if cuda else torch.float32, device=0 if cuda else -1)
    gen = {"language": language, "task": "transcribe"} if language else {"task": "transcribe"}
    out = asr(paths, batch_size=batch_size, chunk_length_s=30, generate_kwargs=gen)
    return [o["text"].strip() for o in out], {"language": language, "decoding": "greedy", "chunk_length_s": 30,
                                              "dtype": "float16" if cuda else "float32"}


ADAPTERS = {"nemo": nemo, "whisper": whisper}
