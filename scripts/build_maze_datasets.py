from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset.build_maze_unique_dataset import DataProcessConfig, preprocess_data


def build_unique(train_samples: int, test_samples: int, seed: int) -> None:
    preprocess_data(
        DataProcessConfig(
            output_dir="data/maze-30x30-unique-1k",
            grid_size=30,
            train_samples=train_samples,
            test_samples=test_samples,
            maze_mode="perfect",
            length_distribution="uniform",
            min_path_length=20,
            max_path_length=300,
            strict_length=False,
            require_unique=True,
            dedupe=True,
            seed=seed,
        )
    )


def build_multi(train_samples: int, test_samples: int, seed: int) -> None:
    # Multi-solution mazes are deliberately more open than the unique/perfect
    # mazes. This keeps generation practical while preserving the ambiguity:
    # start/goal pairs must still have at least two shortest paths.
    preprocess_data(
        DataProcessConfig(
            output_dir="data/maze-30x30-multi-1k",
            grid_size=30,
            train_samples=train_samples,
            test_samples=test_samples,
            maze_mode="random",
            wall_prob=0.08,
            length_distribution="uniform",
            min_path_length=20,
            max_path_length=80,
            strict_length=False,
            require_multiple=True,
            dedupe=True,
            max_length_resamples=50,
            max_grid_attempts=100,
            max_start_attempts=200,
            seed=seed,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["unique", "multi", "both"], default="both")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.which in {"unique", "both"}:
        build_unique(args.train_samples, args.test_samples, args.seed)
    if args.which in {"multi", "both"}:
        build_multi(args.train_samples, args.test_samples, args.seed + 1)


if __name__ == "__main__":
    main()
