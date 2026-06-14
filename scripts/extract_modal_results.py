#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from google.protobuf.message import DecodeError
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


DEFAULT_CANDIDATES = (
    Path("modal_outputs/outputs"),
    Path("modal_outputs/outputs_smoke"),
    Path("modal_outputs/outputs/outputs"),
)


def find_root(path_arg: str | None) -> Path:
    if path_arg:
        root = Path(path_arg)
        if not root.exists():
            raise FileNotFoundError(root)
        return root
    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find modal outputs root.")


def json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def history_key(item: Any) -> str:
    if getattr(item, "key", ""):
        return item.key
    nested = list(getattr(item, "nested_key", []))
    return "/".join(nested)


def read_wandb_history(run_dir: Path) -> dict[str, Any]:
    wandb_files = sorted(run_dir.glob("wandb/offline-run-*/run-*.wandb"))
    if not wandb_files:
        return {}

    ds = DataStore()
    ds.open_for_scan(str(wandb_files[0]))
    last: dict[str, Any] = {}
    try:
        while True:
            record = ds.scan_record()
            if record is None:
                break

            data = record[1] if isinstance(record, tuple) else record
            parsed = wandb_internal_pb2.Record()
            try:
                parsed.ParseFromString(data)
            except DecodeError:
                continue

            if parsed.WhichOneof("record_type") != "history":
                continue

            for item in parsed.history.item:
                key = history_key(item)
                if key:
                    last[key] = json_value(item.value_json)
    finally:
        ds.close()
    return last


def read_config(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "all_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id_dir in sorted(root.iterdir()):
        if not run_id_dir.is_dir():
            continue
        for run_dir in sorted(run_id_dir.glob("*/*")):
            if not (run_dir / "all_config.yaml").exists():
                continue

            cfg = read_config(run_dir)
            hist = read_wandb_history(run_dir)
            arch = cfg.get("arch", {})
            wandb_meta = cfg.get("wandb_meta", {})
            ckpts = sorted((run_dir / "checkpoints").glob("*.pth"))

            rows.append(
                {
                    "run_id": run_id_dir.name,
                    "run_name": wandb_meta.get("name"),
                    "model": arch.get("short_name") or arch.get("name"),
                    "arch_name": arch.get("name"),
                    "max_steps": cfg.get("max_steps"),
                    "params": hist.get("num_params"),
                    "step": hist.get("step"),
                    "train_accuracy": hist.get("train/accuracy"),
                    "train_exact_accuracy": hist.get("train/exact_accuracy"),
                    "train_lm_loss": hist.get("train/lm_loss"),
                    "train_total_loss": hist.get("train/total_loss"),
                    "train_gate_mean": hist.get("train/gate_mean"),
                    "train_hard_gate_mean": hist.get("train/hard_gate_mean"),
                    "train_library_entropy": hist.get("train/library_entropy"),
                    "checkpoint": str(ckpts[-1]) if ckpts else "",
                    "run_dir": str(run_dir),
                }
            )
    rows.sort(key=lambda r: (int(r["max_steps"] or 0), r["run_name"] or ""))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        "run_name",
        "model",
        "step",
        "params",
        "train_accuracy",
        "train_exact_accuracy",
        "train_lm_loss",
        "train_total_loss",
        "train_gate_mean",
        "train_hard_gate_mean",
        "train_library_entropy",
    ]
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Results root, e.g. modal_outputs/outputs_smoke")
    parser.add_argument("--out", default="modal_results_summary.csv")
    args = parser.parse_args()

    root = find_root(args.root)
    rows = collect_rows(root)
    if not rows:
        raise RuntimeError(f"No runs found under {root}")

    out = Path(args.out)
    write_csv(rows, out)
    print(f"Using results root: {root}")
    print(f"Wrote {out}\n")
    print_table(rows)

    ranked = [row for row in rows if isinstance(row.get("train_accuracy"), (int, float))]
    if ranked:
        best = max(ranked, key=lambda row: row["train_accuracy"])
        print(
            f"\nBest train_accuracy: {best['run_name']} "
            f"({best['train_accuracy']:.6g}, loss={best.get('train_total_loss')})"
        )


if __name__ == "__main__":
    main()
