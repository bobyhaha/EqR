from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal


APP_NAME = "eqr-lg-prm"
REMOTE_REPO = "/root/EqR_Modified"
RESULTS_VOLUME = "eqr-lg-prm-results"
DATA_VOLUME = "eqr-lg-prm-data"
GPU_FALLBACKS = ["B200", "H200", "H100"]
BUILD_GPU = "B200"

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
        "numpy",
        "pydantic",
        "omegaconf",
        "hydra-core",
        "wandb",
        "coolname",
        "psutil",
        "tqdm",
        "pyyaml",
        "argdantic",
        "colorama",
        "huggingface_hub",
        "matplotlib",
        "plotly",
        "pandas",
        "scipy",
        "scikit-learn",
        "packaging",
        "wheel",
    )
    .pip_install(
        "adam-atan2==0.0.3",
        extra_options="--no-build-isolation --no-cache-dir",
        gpu=BUILD_GPU,
        env={"TORCH_CUDA_ARCH_LIST": "9.0"},
    )
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation --no-cache-dir",
        gpu=BUILD_GPU,
        env={"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8"},
    )
    .env(
        {
            "OUTPUT_ROOT": "/outputs",
            "RUN_ROOT": "/outputs",
            "WANDB_MODE": "offline",
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
    if env:
        run_env.update(env)
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, env=run_env, check=True)


def _ensure_sudoku_data() -> None:
    expected = Path(REMOTE_REPO) / "data" / "sudoku-extreme-1k-aug-1000" / "train"
    if expected.exists():
        print(f"Sudoku data already present: {expected}", flush=True)
        return
    _run(["bash", "scripts/download_artifacts.sh"])
    data_volume.commit()


def _train_overrides(max_steps: Optional[int], smoke: bool, disable_compile: bool) -> list[str]:
    overrides = ["wandb_mode=offline"]
    if max_steps is not None:
        overrides.append(f"max_steps={max_steps}")
    if smoke:
        overrides.extend(
            [
                "max_steps=2",
                "eval_interval_steps=null",
                "checkpoint_interval_steps=null",
                "heavy_metrics_log_interval=null",
                "steps_hist_log_interval_steps=null",
            ]
        )
    if disable_compile:
        overrides.append("gradient_checkpoint=false")
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

    _run(["nvidia-smi"])
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    _run(["python", "-m", "py_compile", "models/lg_prm.py", "scripts/print_model_params.py"])
    _run(["python", "scripts/print_model_params.py", config])

    _ensure_sudoku_data()
    _run(
        ["bash", "scripts/train.sh", config, *_train_overrides(None, smoke=True, disable_compile=True)],
        env={"DISABLE_COMPILE": "1"},
    )
    results_volume.commit()
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
    _run(["nvidia-smi"])
    _run(["python", "scripts/print_model_params.py", config])
    _ensure_sudoku_data()
    env = {"DISABLE_COMPILE": "1"} if disable_compile else None
    _run(["bash", "scripts/train.sh", config, *_train_overrides(max_steps, smoke=False, disable_compile=disable_compile)], env=env)
    results_volume.commit()
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
    _run(["nvidia-smi"])
    _run(["python", "scripts/print_model_params.py"])
    _ensure_sudoku_data()
    env = {"DISABLE_COMPILE": "1"} if disable_compile else None
    for config in SUDOKU_CONFIGS:
        _run(["bash", "scripts/train.sh", config, *_train_overrides(max_steps, smoke=False, disable_compile=disable_compile)], env=env)
        results_volume.commit()
    return {"status": "ok", "configs": SUDOKU_CONFIGS, "results_volume": RESULTS_VOLUME}


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    config: str = "lg_prm_noisy_soft_sudoku",
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
) -> None:
    if mode == "all":
        result = train_all.remote(max_steps=max_steps, disable_compile=disable_compile)
    elif mode == "smoke":
        result = smoke.remote(config=config)
    elif mode == "train":
        result = train_config.remote(config=config, max_steps=max_steps, disable_compile=disable_compile)
    else:
        raise ValueError("mode must be one of: smoke, train, all")

    print(result)
    print()
    print("Results are persisted in Modal Volume:", RESULTS_VOLUME)
    print("List results:")
    print(f"  modal volume ls {RESULTS_VOLUME} outputs")
    print("Download results:")
    print(f"  modal volume get {RESULTS_VOLUME} outputs ./modal_outputs")
