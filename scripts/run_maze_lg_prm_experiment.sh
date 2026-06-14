#!/usr/bin/env bash
# Run the maze comparison on unique-solution and multi-solution datasets.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/build_maze_datasets.py --which both

configs=(
  eqr_maze_unique
  trm_maze_unique
  lg_prm_hard_maze_unique
  lg_prm_soft_maze_unique
  lg_prm_noisy_hard_maze_unique
  lg_prm_noisy_soft_maze_unique
  lg_prm_no_library_maze_unique
  eqr_maze_multi
  trm_maze_multi
  lg_prm_hard_maze_multi
  lg_prm_soft_maze_multi
  lg_prm_noisy_hard_maze_multi
  lg_prm_noisy_soft_maze_multi
  lg_prm_no_library_maze_multi
)

python scripts/print_model_params.py "${configs[@]}"

for cfg in "${configs[@]}"; do
  echo "===== training ${cfg} ====="
  bash scripts/train.sh "${cfg}"
done
