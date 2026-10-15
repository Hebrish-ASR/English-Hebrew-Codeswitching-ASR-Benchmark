"""Run the benchmark on a Modal GPU and write the result file locally.

    MODAL_PROFILE=danielrosehill modal run modal_app.py --model nemo:nvidia/parakeet-tdt-0.6b-v2 --name "Parakeet TDT 0.6B v2"

Any `ehcs-bench run` flag can be passed through --args, as one string. Nothing persists on
Modal: no Volume, weights are downloaded per run (a deliberate cost choice, 2026-10-15).
The image is the NeMo 2.7.3 build proven by the Parakeet fine-tune; the pins exist because
unpinned installs broke the TDT numba kernel twice (see that repo's training/app.py).
"""
import json
import shlex
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
PINS = ["torch==2.11.0", "numba==0.65.0", "numba-cuda[cu12]==0.30.0"]
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential")
    .pip_install("Cython", "packaging")
    .pip_install(*PINS, index_url="https://download.pytorch.org/whl/cu128", extra_index_url="https://pypi.org/simple")
    .pip_install("nemo_toolkit[asr,cu12]==2.7.3", *PINS, "jiwer==3.1.0", "soundfile", "librosa", "huggingface_hub")
    .add_local_python_source("ehcs_bench")
)
app = modal.App("ehcs-asr-benchmark")


@app.function(image=image, gpu="L40S", timeout=3600, secrets=[modal.Secret.from_name("parakeet-finetune-hf")])
def run(argv: list[str]) -> str:
    from ehcs_bench.cli import main
    out = "/tmp/result.json"
    main(["run", *argv, "--out", out, "--host", "modal:L40S"])
    return Path(out).read_text()


@app.local_entrypoint()
def entry(model: str, name: str = "", args: str = "", out: str = ""):
    argv = ["--model", model] + (["--name", name] if name else []) + shlex.split(args)
    res = run.remote(argv)
    j = json.loads(res)
    from ehcs_bench.cli import slug
    p = Path(out or ROOT / "results" / f"{slug(j['model']['name'])}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(res)
    c = j["metrics"]["subsets"]["clean"]
    print(f"{j['model']['name']}: clean WER {c['wer']:.2%}, term recall {c['terms']['all']['recall']:.1%} -> {p}")
