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


def _model_config(arch: dict) -> dict:
    cfg = {k: v for k, v in arch.items() if k not in {"name", "short_name", "loss"}}
    cfg.update(
        {
            "batch_size": 2,
            "seq_len": 81,
            "vocab_size": 11,
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
            "lg_prm_soft_sudoku",
            "lg_prm_noisy_hard_sudoku",
            "lg_prm_noisy_soft_sudoku",
            "lg_prm_no_library_sudoku",
        ],
    )
    args = ap.parse_args()
    for name in args.configs:
        arch = _arch_from_train_config(name)
        cls = load_model_class(arch["name"])
        model = cls(_model_config(arch))
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:28s} {arch.get('short_name', arch['name']):18s} params={params:,}")


if __name__ == "__main__":
    main()
