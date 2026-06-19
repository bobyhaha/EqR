from __future__ import annotations

import os
import json
import time
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import modal
import yaml


APP_NAME = "eqr-lg-prm"
REMOTE_REPO = "/root/EqR_Modified"
RESULTS_VOLUME = "eqr-lg-prm-results"
DATA_VOLUME = "eqr-lg-prm-data"
# torch==2.6.0+cu124 supports Hopper (H100/H200) but not Blackwell B200 sm_100.
# Keep B200 out of runtime fallback until the image moves to a PyTorch/CUDA build
# that explicitly supports sm_100.
GPU_FALLBACKS = ["H200", "H100"]
GPU5_FALLBACKS = ["H200:5", "H100:5"]
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
    "lg_prm_concat_sudoku",
    "lg_prm_soft_sudoku",
    "lg_prm_noisy_hard_sudoku",
    "lg_prm_noisy_soft_sudoku",
    "lg_prm_no_library_sudoku",
]

MAZE_UNIQUE_CONFIGS = [
    "eqr_maze_unique",
    "trm_maze_unique",
    "lg_prm_hard_maze_unique",
    "lg_prm_concat_maze_unique",
    "lg_prm_soft_maze_unique",
    "lg_prm_noisy_hard_maze_unique",
    "lg_prm_noisy_soft_maze_unique",
    "lg_prm_no_library_maze_unique",
]

MAZE_MULTI_CONFIGS = [
    "eqr_maze_multi",
    "trm_maze_multi",
    "lg_prm_hard_maze_multi",
    "lg_prm_concat_maze_multi",
    "lg_prm_soft_maze_multi",
    "lg_prm_noisy_hard_maze_multi",
    "lg_prm_noisy_soft_maze_multi",
    "lg_prm_no_library_maze_multi",
]

MAZE_CONFIGS = MAZE_UNIQUE_CONFIGS + MAZE_MULTI_CONFIGS
ALL_CONFIGS = SUDOKU_CONFIGS + MAZE_CONFIGS


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
            "modal_outputs",
            "modal_outputs_*",
            "eqr_modal_outputs*.tar.gz",
            "modal_results_summary.csv",
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


def _run_with_periodic_result_commits(
    cmd: list[str],
    *,
    env: Optional[dict[str, str]] = None,
    commit_interval_seconds: int = 60,
) -> None:
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
    proc = subprocess.Popen(cmd, cwd=REMOTE_REPO, env=run_env)
    last_commit = time.monotonic()
    while True:
        ret = proc.poll()
        now = time.monotonic()
        if ret is not None:
            results_volume.commit()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
            return
        if now - last_commit >= commit_interval_seconds:
            print("[Modal] committing results volume during training", flush=True)
            results_volume.commit()
            last_commit = now
        time.sleep(5)


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


def _split_has_samples(dataset_root: Path, split: str, min_samples: int) -> bool:
    metadata_path = dataset_root / split / "dataset.json"
    if not metadata_path.exists():
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata.get("total_samples", 0)) >= min_samples


def _maze_dataset_ready(dataset_root: Path, min_samples: int, expected_mode: Optional[str] = None) -> bool:
    if expected_mode is not None:
        config_path = dataset_root / "generation_config.json"
        if not config_path.exists():
            return False
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                generation_config = json.load(handle)
        except Exception:
            return False
        if generation_config.get("maze_mode") != expected_mode:
            return False
    return _split_has_samples(dataset_root, "train", min_samples) and _split_has_samples(dataset_root, "test", min_samples)


def _ensure_maze_data(which: str = "both", *, smoke: bool = False) -> None:
    expected_unique = Path(REMOTE_REPO) / "data" / "maze-30x30-unique-1k"
    expected_multi = Path(REMOTE_REPO) / "data" / "maze-30x30-multi-1k"
    expected = {"unique": (expected_unique, "perfect"), "multi": (expected_multi, "direct_multi")}
    if which not in {"unique", "multi", "both"}:
        raise ValueError(f"unknown maze dataset selector: {which}")

    required = list(expected.values()) if which == "both" else [expected[which]]
    min_samples = 4 if smoke else 1000
    if all(_maze_dataset_ready(path, min_samples, mode) for path, mode in required):
        print(f"Maze data already present for {which}: {', '.join(str(path) for path, _ in required)}", flush=True)
        return

    cmd = ["python", "scripts/build_maze_datasets.py", "--which", which]
    if smoke:
        cmd.extend(["--train-samples", "4", "--test-samples", "4"])
    _run(cmd)
    data_volume.commit()


def _ensure_data_for_config(config: str, *, smoke: bool = False) -> None:
    if "sudoku" in config:
        _ensure_sudoku_data()
    elif "maze_unique" in config:
        _ensure_maze_data("unique", smoke=smoke)
    elif "maze_multi" in config:
        _ensure_maze_data("multi", smoke=smoke)
    elif "maze" in config:
        _ensure_maze_data("both", smoke=smoke)
    else:
        raise ValueError(f"Cannot infer dataset for config: {config}")


def _configs_from(configs: list[str], start_config: Optional[str]) -> list[str]:
    if not start_config:
        return configs
    if start_config not in configs:
        raise ValueError(f"start_config={start_config!r} is not in this run list: {configs}")
    return configs[configs.index(start_config):]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_name_for_config(config: str) -> str:
    train = _load_yaml(Path(REMOTE_REPO) / "config" / "train" / f"{config}.yaml")
    arch_name = None
    dataset_name = None
    for item in train.get("defaults", []):
        if isinstance(item, dict):
            arch_name = item.get("/arch", arch_name)
            dataset_name = item.get("/dataset", dataset_name)
    if not arch_name or not dataset_name:
        raise ValueError(f"Could not resolve arch/dataset for {config}")
    arch = _load_yaml(Path(REMOTE_REPO) / "config" / "arch" / f"{arch_name}.yaml")
    dataset = _load_yaml(Path(REMOTE_REPO) / "config" / "dataset" / f"{dataset_name}.yaml")
    model_id = arch.get("short_name") or str(arch["name"]).split("@")[-1]
    return f"{model_id}-{dataset['name']}"


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_"):
        return -1
    try:
        return int(stem.split("_", 2)[1])
    except Exception:
        return -1


def _latest_checkpoint_for_config(config: str, experiment_name: Optional[str]) -> Optional[Path]:
    outputs = Path("/outputs/outputs")
    if not outputs.exists():
        return None

    expected_name = _run_name_for_config(config)
    candidates: list[Path] = []
    for cfg_path in outputs.glob("*/*/*/all_config.yaml"):
        try:
            saved = _load_yaml(cfg_path)
        except Exception:
            continue
        meta = saved.get("wandb_meta", {}) or {}
        if meta.get("name") != expected_name:
            continue
        if experiment_name and meta.get("project") != experiment_name:
            continue
        candidates.extend(cfg_path.parent.glob("checkpoints/step_*.pth"))

    if not candidates:
        return None
    return max(candidates, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))


def _train_overrides(
    max_steps: Optional[int],
    smoke: bool,
    disable_compile: bool,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = None,
    load_checkpoint: Optional[Path] = None,
) -> list[str]:
    overrides = ["++wandb_mode=offline"]
    if experiment_name:
        overrides.append(f"++project_name={experiment_name}")
    if load_checkpoint is not None:
        overrides.append(f"++load_checkpoint={load_checkpoint}")
    if max_steps is not None:
        overrides.append(f"++max_steps={max_steps}")
    if checkpoint_interval_steps is not None:
        overrides.append(f"++checkpoint_interval_steps={checkpoint_interval_steps}")
    if skip_eval:
        overrides.append("++eval_interval_steps=null")
    if smoke:
        overrides.extend(
            [
                "++max_steps=1",
                "++global_batch_size=4",
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
                "scripts/build_maze_datasets.py",
                "scripts/print_model_params.py",
            ]
        )
        _run(["python", "-c", "import exceptiongroup, adam_atan2; print('deps ok')"])
        _run(["python", "scripts/print_model_params.py", config])

        _ensure_data_for_config(config, smoke=True)
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
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, str]:
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", config])
        _ensure_data_for_config(config)
        env = {"DISABLE_COMPILE": "1"} if disable_compile else None
        checkpoint = _latest_checkpoint_for_config(config, experiment_name) if auto_resume else None
        if checkpoint is not None:
            print(f"Resuming {config} from {checkpoint}", flush=True)
        _run_with_periodic_result_commits(
            [
                "bash",
                "scripts/train.sh",
                config,
                *_train_overrides(
                    max_steps,
                    smoke=False,
                    disable_compile=disable_compile,
                    skip_eval=skip_eval,
                    experiment_name=experiment_name,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                    load_checkpoint=checkpoint,
                ),
            ],
            env=env,
            commit_interval_seconds=60,
        )
    finally:
        _commit_volumes()
    return {"status": "ok", "config": config, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU5_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_config_5gpu(
    config: str,
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", config])
        _ensure_data_for_config(config)
        env = {"NPROC_PER_NODE": "5"}
        if disable_compile:
            env["DISABLE_COMPILE"] = "1"
        checkpoint = _latest_checkpoint_for_config(config, experiment_name) if auto_resume else None
        if checkpoint is not None:
            print(f"Resuming {config} from {checkpoint}", flush=True)
        _run_with_periodic_result_commits(
            [
                "bash",
                "scripts/train.sh",
                config,
                *_train_overrides(
                    max_steps,
                    smoke=False,
                    disable_compile=disable_compile,
                    skip_eval=skip_eval,
                    experiment_name=experiment_name,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                    load_checkpoint=checkpoint,
                ),
                "++global_batch_size=770",
            ],
            env=env,
            commit_interval_seconds=60,
        )
    finally:
        _commit_volumes()
    return {"status": "ok", "config": config, "gpus": 5, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU8_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_config_8gpu(
    config: str,
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", config])
        _ensure_data_for_config(config)
        env = {"NPROC_PER_NODE": "8"}
        if disable_compile:
            env["DISABLE_COMPILE"] = "1"
        checkpoint = _latest_checkpoint_for_config(config, experiment_name) if auto_resume else None
        if checkpoint is not None:
            print(f"Resuming {config} from {checkpoint}", flush=True)
        _run_with_periodic_result_commits(
            [
                "bash",
                "scripts/train.sh",
                config,
                *_train_overrides(
                    max_steps,
                    smoke=False,
                    disable_compile=disable_compile,
                    skip_eval=skip_eval,
                    experiment_name=experiment_name,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                    load_checkpoint=checkpoint,
                ),
            ],
            env=env,
            commit_interval_seconds=60,
        )
    finally:
        _commit_volumes()
    return {"status": "ok", "config": config, "gpus": 8, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


def _train_configs(
    configs: list[str],
    max_steps: Optional[int],
    disable_compile: bool,
    env: Optional[dict[str, str]],
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> None:
    if len(configs) > 1:
        _run(["python", "scripts/check_param_match.py", *configs, "--max-relative-gap", "0.02"])
    for config in configs:
        _ensure_data_for_config(config)
        checkpoint = _latest_checkpoint_for_config(config, experiment_name) if auto_resume else None
        if checkpoint is not None:
            print(f"Resuming {config} from {checkpoint}", flush=True)
        _run_with_periodic_result_commits(
            [
                "bash",
                "scripts/train.sh",
                config,
                *_train_overrides(
                    max_steps,
                    smoke=False,
                    disable_compile=disable_compile,
                    skip_eval=skip_eval,
                    experiment_name=experiment_name,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                    load_checkpoint=checkpoint,
                ),
            ],
            env=env,
            commit_interval_seconds=60,
        )
        _commit_volumes()


@app.function(
    image=image,
    gpu=GPU_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_all(
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    configs = _configs_from(SUDOKU_CONFIGS, start_config)
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", *configs])
        env = {"DISABLE_COMPILE": "1"} if disable_compile else None
        _train_configs(configs, max_steps, disable_compile, env, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": configs, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_maze(
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    configs = _configs_from(MAZE_CONFIGS, start_config)
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", *configs])
        env = {"DISABLE_COMPILE": "1"} if disable_compile else None
        _train_configs(configs, max_steps, disable_compile, env, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": configs, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_all_datasets(
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    configs = _configs_from(ALL_CONFIGS, start_config)
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", *configs])
        env = {"DISABLE_COMPILE": "1"} if disable_compile else None
        _train_configs(configs, max_steps, disable_compile, env, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": configs, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU8_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_all_8gpu(
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    configs = _configs_from(SUDOKU_CONFIGS, start_config)
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", *configs])
        env = {"NPROC_PER_NODE": "8"}
        if disable_compile:
            env["DISABLE_COMPILE"] = "1"
        _train_configs(configs, max_steps, disable_compile, env, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": configs, "gpus": 8, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU8_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_maze_8gpu(
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    configs = _configs_from(MAZE_CONFIGS, start_config)
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", *configs])
        env = {"NPROC_PER_NODE": "8"}
        if disable_compile:
            env["DISABLE_COMPILE"] = "1"
        _train_configs(configs, max_steps, disable_compile, env, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": configs, "gpus": 8, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


@app.function(
    image=image,
    gpu=GPU8_FALLBACKS,
    timeout=60 * 60 * 24,
    volumes={
        "/outputs": results_volume,
        f"{REMOTE_REPO}/data": data_volume,
    },
)
def train_all_datasets_8gpu(
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> dict[str, object]:
    configs = _configs_from(ALL_CONFIGS, start_config)
    try:
        _run(["nvidia-smi"])
        _run(["python", "scripts/print_model_params.py", *configs])
        env = {"NPROC_PER_NODE": "8"}
        if disable_compile:
            env["DISABLE_COMPILE"] = "1"
        _train_configs(configs, max_steps, disable_compile, env, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    finally:
        _commit_volumes()
    return {"status": "ok", "configs": configs, "gpus": 8, "experiment_name": experiment_name, "results_volume": RESULTS_VOLUME}


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


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={"/outputs": results_volume},
)
def summarize_results(dataset_filter: Optional[str] = None) -> list[dict[str, object]]:
    rows = []
    root = Path("/outputs/outputs")
    if not root.exists():
        print("No outputs directory found", flush=True)
        return rows

    for cfg_path in sorted(root.glob("*/*/*/all_config.yaml")):
        try:
            cfg = _load_yaml(cfg_path)
        except Exception as exc:
            print(f"Skipping unreadable config {cfg_path}: {exc}", flush=True)
            continue

        meta = cfg.get("wandb_meta", {}) or {}
        dataset = (cfg.get("dataset", {}) or {}).get("name", "")
        run_name = meta.get("name", "")
        if dataset_filter:
            needle = dataset_filter.lower()
            if needle not in str(dataset).lower() and needle not in str(run_name).lower():
                continue

        steps = []
        for ckpt in cfg_path.parent.glob("checkpoints/step_*.pth"):
            step = _checkpoint_step(ckpt)
            if step >= 0:
                steps.append(step)

        row = {
            "run_id": cfg_path.parts[3],
            "run_name": run_name,
            "project": meta.get("project", ""),
            "dataset": dataset,
            "params": cfg.get("num_params"),
            "max_step": max(steps) if steps else None,
            "run_dir": str(cfg_path.parent),
        }
        rows.append(row)

    print(
        f"{'run_id':10s} {'max_step':>8s} {'params':>12s} {'project':28s} run_name",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['run_id']:10s} {str(row['max_step']):>8s} "
            f"{str(row['params']):>12s} {str(row['project'])[:28]:28s} {row['run_name']}",
            flush=True,
        )
    return rows


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={"/outputs": results_volume},
)
def pack_sudoku_finals(
    archive_name: str = "sudoku_final_checkpoints.tar.gz",
    project_filter: Optional[str] = "all-datasets-all-models",
) -> dict[str, object]:
    root = Path("/outputs/outputs")
    dst = Path("/outputs") / archive_name
    if dst.exists():
        dst.unlink()

    selected: dict[str, tuple[int, Path, Path]] = {}
    for cfg_path in sorted(root.glob("*/*/*/all_config.yaml")):
        cfg = _load_yaml(cfg_path)
        meta = cfg.get("wandb_meta", {}) or {}
        run_name = str(meta.get("name", ""))
        project = meta.get("project")
        if "Sudoku" not in run_name:
            continue
        if project_filter is not None and project != project_filter:
            continue
        ckpts = [p for p in cfg_path.parent.glob("checkpoints/step_*.pth") if _checkpoint_step(p) >= 0]
        if not ckpts:
            continue
        ckpt = max(ckpts, key=lambda p: _checkpoint_step(p))
        step = _checkpoint_step(ckpt)
        old = selected.get(run_name)
        if old is None or step > old[0]:
            selected[run_name] = (step, cfg_path, ckpt)

    if not selected:
        raise FileNotFoundError(f"No Sudoku final checkpoints found for project={project_filter!r}")

    with tarfile.open(dst, "w:gz") as tar:
        manifest = []
        for run_name, (step, cfg_path, ckpt) in sorted(selected.items()):
            run_dir = cfg_path.parent
            run_id = run_dir.parts[3]
            safe_name = run_name.replace("/", "_").replace(" ", "_")
            prefix = f"{run_id}_{safe_name}"
            tar.add(cfg_path, arcname=f"{prefix}/all_config.yaml")
            tar.add(ckpt, arcname=f"{prefix}/{ckpt.name}")
            for wandb_file in run_dir.glob("wandb/offline-run-*/*.wandb"):
                tar.add(wandb_file, arcname=f"{prefix}/wandb/{wandb_file.name}")
            for log_file in run_dir.glob("wandb/offline-run-*/files/output.log"):
                tar.add(log_file, arcname=f"{prefix}/wandb/output.log")
            manifest.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "step": step,
                    "checkpoint": str(ckpt),
                    "run_dir": str(run_dir),
                }
            )

        manifest_path = Path("/tmp/sudoku_final_manifest.json")
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        tar.add(manifest_path, arcname="manifest.json")

    results_volume.commit()
    print(f"Packed {len(selected)} Sudoku final runs into {archive_name}", flush=True)
    for item in manifest:
        print(f"{item['run_id']:10s} step={item['step']:>6} {item['run_name']}", flush=True)
    return {
        "status": "ok",
        "archive": archive_name,
        "runs": manifest,
        "download": f"modal volume get {RESULTS_VOLUME} {archive_name} ./{archive_name}",
    }


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    config: str = "lg_prm_noisy_soft_sudoku",
    max_steps: Optional[int] = None,
    disable_compile: bool = False,
    start_config: Optional[str] = None,
    skip_eval: bool = False,
    experiment_name: Optional[str] = None,
    checkpoint_interval_steps: Optional[int] = 1000,
    auto_resume: bool = True,
) -> None:
    if mode == "all":
        result = train_all.remote(max_steps=max_steps, disable_compile=disable_compile, start_config=start_config, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode == "all8":
        result = train_all_8gpu.remote(max_steps=max_steps, disable_compile=disable_compile, start_config=start_config, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode == "maze":
        result = train_maze.remote(max_steps=max_steps, disable_compile=disable_compile, start_config=start_config, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode == "maze8":
        result = train_maze_8gpu.remote(max_steps=max_steps, disable_compile=disable_compile, start_config=start_config, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode in {"all-datasets", "all_datasets"}:
        result = train_all_datasets.remote(max_steps=max_steps, disable_compile=disable_compile, start_config=start_config, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode in {"all-datasets8", "all_datasets8"}:
        result = train_all_datasets_8gpu.remote(max_steps=max_steps, disable_compile=disable_compile, start_config=start_config, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode in {"pack", "pack-results"}:
        result = pack_results.remote()
    elif mode in {"summary", "summarize", "summarize-results"}:
        result = summarize_results.remote(dataset_filter=config if config != "lg_prm_noisy_soft_sudoku" else None)
    elif mode in {"pack-sudoku-finals", "pack_sudoku_finals"}:
        result = pack_sudoku_finals.remote()
    elif mode == "smoke":
        result = smoke.remote(config=config)
    elif mode == "train":
        result = train_config.remote(config=config, max_steps=max_steps, disable_compile=disable_compile, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode == "train5":
        result = train_config_5gpu.remote(config=config, max_steps=max_steps, disable_compile=disable_compile, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    elif mode == "train8":
        result = train_config_8gpu.remote(config=config, max_steps=max_steps, disable_compile=disable_compile, skip_eval=skip_eval, experiment_name=experiment_name, checkpoint_interval_steps=checkpoint_interval_steps, auto_resume=auto_resume)
    else:
        raise ValueError("mode must be one of: smoke, train, train5, train8, all, all8, maze, maze8, all-datasets, all-datasets8, pack, summarize-results, pack-sudoku-finals")

    print(result)
    print()
    print("Results are persisted in Modal Volume:", RESULTS_VOLUME)
    print("List results:")
    print(f"  modal volume ls {RESULTS_VOLUME} outputs")
    print("Download results:")
    print(f"  modal volume get {RESULTS_VOLUME} outputs ./modal_outputs")
