import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.functions import load_model_class


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _arch_from_train_config(name: str) -> dict:
    train = _load_yaml(os.path.join("config", "train", f"{name}.yaml"))
    arch_name = None
    for item in train.get("defaults", []):
        if isinstance(item, dict) and "/arch" in item:
            arch_name = item["/arch"]
    if arch_name is None:
        raise ValueError(f"Could not find /arch default for {name}")
    arch = _load_yaml(os.path.join("config", "arch", f"{arch_name}.yaml"))
    arch.update(train.get("arch", {}) or {})
    return arch


def _dataset_from_train_config(name: str) -> dict:
    train = _load_yaml(os.path.join("config", "train", f"{name}.yaml"))
    dataset_name = None
    for item in train.get("defaults", []):
        if isinstance(item, dict) and "/dataset" in item:
            dataset_name = item["/dataset"]
    if dataset_name is None:
        raise ValueError(f"Could not find /dataset default for {name}")
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
    cfg.update({"batch_size": 2, "seq_len": seq_len, "vocab_size": vocab_size})
    return cfg


def _group_name(config: str) -> str:
    if "sudoku" in config:
        return "sudoku"
    if "maze_unique" in config:
        return "maze_unique"
    if "maze_multi" in config:
        return "maze_multi"
    return "other"


def _num_params(config: str) -> tuple[str, int]:
    arch = _arch_from_train_config(config)
    dataset = _dataset_from_train_config(config)
    model_cls = load_model_class(arch["name"])
    model = model_cls(_model_config(arch, dataset))
    return arch.get("short_name", arch["name"]), sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--max-relative-gap", type=float, default=0.02)
    args = parser.parse_args()

    grouped: dict[str, list[tuple[str, str, int]]] = {}
    for config in args.configs:
        model_name, params = _num_params(config)
        grouped.setdefault(_group_name(config), []).append((config, model_name, params))

    failed = False
    for group, rows in grouped.items():
        if len(rows) < 2:
            continue
        min_params = min(params for _, _, params in rows)
        max_params = max(params for _, _, params in rows)
        relative_gap = (max_params - min_params) / max_params
        print(f"{group}: min={min_params:,} max={max_params:,} rel_gap={relative_gap:.4%}")
        for config, model_name, params in rows:
            print(f"  {config:32s} {model_name:18s} params={params:,}")
        if relative_gap > args.max_relative_gap:
            failed = True

    if failed:
        raise SystemExit(f"Parameter mismatch exceeds {args.max_relative_gap:.2%}")


if __name__ == "__main__":
    main()
