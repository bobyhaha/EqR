import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.functions import load_model_class


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _arch_from_train_config(name: str) -> dict:
    train_path = os.path.join("config", "train", f"{name}.yaml")
    train = _load_yaml(train_path)
    arch_name = None
    for item in train.get("defaults", []):
        if isinstance(item, dict) and "/arch" in item:
            arch_name = item["/arch"]
    if arch_name is None:
        raise ValueError(f"Could not find /arch default in {train_path}")
    arch = _load_yaml(os.path.join("config", "arch", f"{arch_name}.yaml"))
    arch.update(train.get("arch", {}) or {})
    return arch


def _dataset_from_train_config(name: str) -> dict:
    train_path = os.path.join("config", "train", f"{name}.yaml")
    train = _load_yaml(train_path)
    dataset_name = None
    for item in train.get("defaults", []):
        if isinstance(item, dict) and "/dataset" in item:
            dataset_name = item["/dataset"]
    if dataset_name is None:
        raise ValueError(f"Could not find /dataset default in {train_path}")
    return _load_yaml(os.path.join("config", "dataset", f"{dataset_name}.yaml"))


def _dataset_shape(dataset: dict) -> tuple[int, int]:
    data_name = str(dataset.get("name", "")).lower()
    data_path = str(dataset.get("data_path", "")).lower()
    if "maze" in data_name or "maze" in data_path:
        return 30 * 30, 6
    return 9 * 9, 11


def _model_config(arch: dict, dataset: dict) -> dict:
    seq_len, vocab_size = _dataset_shape(dataset)
    cfg = {k: v for k, v in arch.items() if k not in {"name", "short_name", "loss"}}
    cfg.update(
        {
            "batch_size": 2,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
        }
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "configs",
        nargs="*",
        default=[
            "eqr_sudoku",
            "trm_sudoku",
            "lg_prm_hard_sudoku",
            "lg_prm_concat_sudoku",
            "lg_prm_soft_sudoku",
            "lg_prm_noisy_hard_sudoku",
            "lg_prm_noisy_soft_sudoku",
            "lg_prm_no_library_sudoku",
            "eqr_maze_unique",
            "trm_maze_unique",
            "lg_prm_hard_maze_unique",
            "lg_prm_concat_maze_unique",
            "lg_prm_soft_maze_unique",
            "lg_prm_noisy_hard_maze_unique",
            "lg_prm_noisy_soft_maze_unique",
            "lg_prm_no_library_maze_unique",
            "eqr_maze_multi",
            "trm_maze_multi",
            "lg_prm_hard_maze_multi",
            "lg_prm_concat_maze_multi",
            "lg_prm_soft_maze_multi",
            "lg_prm_noisy_hard_maze_multi",
            "lg_prm_noisy_soft_maze_multi",
            "lg_prm_no_library_maze_multi",
        ],
    )
    args = ap.parse_args()
    for name in args.configs:
        arch = _arch_from_train_config(name)
        dataset = _dataset_from_train_config(name)
        cls = load_model_class(arch["name"])
        model = cls(_model_config(arch, dataset))
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:28s} {arch.get('short_name', arch['name']):18s} params={params:,}")


if __name__ == "__main__":
    main()
