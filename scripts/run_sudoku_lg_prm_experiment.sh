#!/usr/bin/env bash
# Run the full Sudoku comparison: EqR, TRM, and five LG-PRM variants.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/print_model_params.py

configs=(
  eqr_sudoku
  trm_sudoku
  lg_prm_hard_sudoku
  lg_prm_soft_sudoku
  lg_prm_noisy_hard_sudoku
  lg_prm_noisy_soft_sudoku
  lg_prm_no_library_sudoku
)

for cfg in "${configs[@]}"; do
  echo "===== training ${cfg} ====="
  bash scripts/train.sh "${cfg}"
done
