#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
OUTPUT_DIR="${OUTPUT_DIR:-../results/mechanistic_cifar}"
DATA_DIR="${DATA_DIR:-../data}"
MODEL="${MODEL:-both}"
PATCH_SIZE="${PATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
LR_SELECTION="${LR_SELECTION:-$PWD/selected_lrs.json}"
if [[ ! -f "$LR_SELECTION" ]]; then
  echo "Missing LR selection: $LR_SELECTION. Run python -u sweep_learning_rates.py first."
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
python -u -m cifar_mech.cli all \
  --model "$MODEL" --patch-size "$PATCH_SIZE" --device "$DEVICE" \
  --lr-selection "$LR_SELECTION" \
  --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --resume \
  2>&1 | tee -a "$OUTPUT_DIR/launch_${MODEL}.log"
