from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import modal


APP_NAME = "eqr-lg-prm"
REMOTE_REPO = "/root/EqR_Modified"
RESULTS_VOLUME = "eqr-lg-prm-results"
DATA_VOLUME = "eqr-lg-prm-data"
# torch==2.6.0+cu124 supports Hopper (H100/H200) but not Blackwell B200 sm_100.
# Keep B200 out of runtime fallback until the image moves to a PyTorch/CUDA build
# that explicitly supports sm_100.
GPU_FALLBACKS = ["H200", "H100"]
GPU8_FALLBACKS = ["H200:8", "H100:8"]
BUILD_GPU = "B200"
CUDA_BUILD_ENV = {
    "CC": "/usr/bin/gcc",
    "CXX": "/usr/bin/g++",
    "TORCH_CUDA_ARCH_LIST": "9.0",
    "MAX_JOBS": "8",
}

SUDOKU_CONFIGS = [
    "eqr_sudoku",
    "trm_sudoku",
    "lg_prm_hard_sudoku",
    "lg_prm_soft_sudoku",
    "lg_prm_noisy_hard_sudoku",
    "lg_prm_noisy_soft_sudoku",
    "lg_prm_no_library_sudoku",
]


app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

repo_dir = Path(__file__).parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "git", "ninja-build", "rsync")
    .pip_install(
        "torch==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy==1.26.4",
        "pydantic==2.11.7",
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
        "wandb==0.18.7",
        "coolname==2.2.0",
        "psutil==6.1.1",
        "tqdm==4.67.1",
        "pyyaml==6.0.2",
        "argdantic==1.3.3",
        "colorama==0.4.6",
        "huggingface_hub==0.27.1",
        "exceptiongroup==1.2.2",
        "matplotlib==3.10.0",
        "plotly==5.24.1",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "scikit-learn==1.6.1",
        "packaging==24.2",
        "wheel==0.45.1",
    )
    .pip_install(
        "adam-atan2==0.0.3",
        extra_options="--no-build-isolation --no-cache-dir",
        gpu=BUILD_GPU,
        env=CUDA_BUILD_ENV,
    )
    .env(
        {
            "OUTPUT_ROOT": "/outputs",
            "RUN_ROOT": "/outputs",
            "WANDB_MODE": "offline",
            "WANDB_CODE_UPLOAD_MODE": "off",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": f"{REMOTE_REPO}/data/.hf-cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        repo_dir,
        remote_path=REMOTE_REPO,
        ignore=[
            ".git",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            "data",
            "downloads",
            "downloaded_checkpoints",
            "outputs",
            "wandb",
            ".venv",
        ],
    )
)


def _run(cmd: list[str], *, env: Optional[dict[str, str]] = None) -> None:
    run_env = os.environ.copy()
    run_env.setdefault("HYDRA_FULL_ERROR", "1")
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("WANDB_MODE", "offline")
    run_env.setdefault("WANDB_CODE_UPLOAD_MODE", "off")
    run_env.setdefault("HF_HOME", f"{REMOTE_REPO}/data/.hf-cache")
    run_env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if env:
        run_env.update(env)
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, env=run_env, check=True)


def _commit_volumes() -> None:
    results_volume.commit()
    data_volume.commit()


def _ensure_sudoku_data() -> None:
    expected = Path(REMOTE_REPO) / "data" / "sudoku-extreme-1k-aug-1000" / "train"
    if expected.exists():
        print(f"Sudoku data already present: {expected}", flush=True)
        return
    _run(["bash", "scripts/download_artifacts.sh"])
    data_volume.commit()


def _train_overrides(max_steps: Optional[int], smoke: bool, disable_compile: bool) -> list[str]:
    overrides = ["++wandb_mode=offline"]
    if max_steps is not None:
        overrides.append(f"++max_steps={max_steps}")
    if smoke:
        overrides.extend(
            [
                "++max_steps=2",
                "++eval_interval_steps=null",
                "++checkpoint_interval_steps=null",
                "++heavy_metrics_log_interval=null",
                "++steps_hist_log_interval_steps=null",
            ]
        )
    if disable_compile:
        overrides.append("++gradient_checkpoint=false")
    return overrides


@app.function(
    image=image,
    gpu=GPU_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def smoke(config: str = "lg_prm_noisy_soft_sudoku") -> dict[str, str]:
    import torch

    try:
        _run(["nvidia-smi"])
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

        _run(
            [
                "python",
                "-m",
                "py_compile",
                "pretrain.py",
                "models/lg_prm.py",
                "utils/wandb.py",
                "utils/checkpoint.py",
                "scripts/print_model_params.py",
            ]
        )
        _run(["python", "-c", "import exceptiongroup, adam_atan2; print('deps ok')"])
        _run(["python", "scripts/print_model_params.py", config])

        _ensure_sudoku_data()
        _run(
            ["bash", "scripts/train.sh", config, *_train_overrides(None, smoke=True, disable_compile=True)],
            env={"DISABLE_COMPILE": "1"},
        )
    finally:
        _commit_volumes()
    return {"status": "ok", "config": config, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_config(
    config: str,
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
) -> dict[str, str]:
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", config])
        _ensure_sudoku_data()
        env = {"DISABLE_COMPILE": "1"} if disable_compile else None
        _run(["bash", "scripts/train.sh", config, *_train_overrides(max_steps, smoke=False, disable_compile=disable_compile)], env=env)
    finally:
        _commit_volumes()
    return {"status": "ok", "config": config, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_all(max_steps: Optional[int] = None, disable_compile: bool = False) -> dict[str, object]:
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py"])
        _ensure_sudoku_data()
        env = {"DISABLE_COMPILE": "1"} if disable_compile else None
        for config in SUDOKU_CONFIGS:
            _run(["bash", "scripts/train.sh", config, *_train_overrides(max_steps, smoke=False, disable_compile=disable_compile)], env=env)
            _commit_volumes()
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": SUDOKU_CONFIGS, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU8_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_all_8gpu(max_steps: Optional[int] = None, disable_compile: bool = False) -> dict[str, object]:
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py"])
        _ensure_sudoku_data()
        env = {"NPROC_PER_NODE": "8"}
        if disable_compile:
            env["DISABLE_COMPILE"] = "1"
        for config in SUDOKU_CONFIGS:
            _run(["bash", "scripts/train.sh", config, *_train_overrides(max_steps, smoke=False, disable_compile=disable_compile)], env=env)
            _commit_volumes()
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": SUDOKU_CONFIGS, "gpus": 8, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={"/outputs": results_volume},
)
def pack_results(archive_name: str = "eqr_modal_outputs.tar.gz") -> dict[str, str]:
    src = Path("/outputs/outputs")
    dst = Path("/outputs") / archive_name
    if not src.exists():
        raise FileNotFoundError(str(src))
    if dst.exists():
        dst.unlink()
    with tarfile.open(dst, "w:gz") as tar:
        tar.add(src, arcname="outputs")
    results_volume.commit()
    return {
        "status": "ok",
        "archive": archive_name,
        "download": f"modal volume get {RESULTS_VOLUME} {archive_name} ./{archive_name}",
    }


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    config: str = "lg_prm_noisy_soft_sudoku",
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
) -> None:
    if mode == "all":
        result = train_all.remote(max_steps=max_steps, disable_compile=disable_compile)
    elif mode == "all8":
        result = train_all_8gpu.remote(max_steps=max_steps, disable_compile=disable_compile)
    elif mode in {"pack", "pack-results"}:
        result = pack_results.remote()
    elif mode == "smoke":
        result = smoke.remote(config=config)
    elif mode == "train":
        result = train_config.remote(config=config, max_steps=max_steps, disable_compile=disable_compile)
    else:
        raise ValueError("mode must be one of: smoke, train, all, all8, pack")

    print(result)
    print()
    print("Results are persisted in Modal Volume:", RESULTS_VOLUME)
    print("List results:")
    print(f"  modal volume ls {RESULTS_VOLUME} outputs")
    print("Download results:")
    print(f"  modal volume get {RESULTS_VOLUME} outputs ./modal_outputs")
