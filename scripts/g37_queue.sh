#!/usr/bin/env bash
# G37 LR x N_A 2D-sweep queue runner for 5090x8 server.
#
# Usage:
#   bash scripts/g37_queue.sh           # run all 18 configs
#   bash scripts/g37_queue.sh 3B        # only 3B (9 configs)
#   bash scripts/g37_queue.sh 14B       # only 14B (9 configs)
#
# Assumes 8x RTX 5090 (32 GB) on the host. 3B fits on 1 GPU; 14B uses 2 GPUs FSDP.

set -euo pipefail

cd "$(dirname "$0")/.."
source setup_env.sh

PYTHON=${PYTHON:-python}
LOG=logs/g37_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs checkpoints

FILTER=${1:-all}

case "$FILTER" in
  3B)  CONFIGS=$(ls configs/g37/g37_3B_*.yaml | sort) ;;
  14B) CONFIGS=$(ls configs/g37/g37_14B_*.yaml | sort) ;;
  *)   CONFIGS=$(ls configs/g37/*.yaml | sort) ;;
esac

run_one() {
  local cfg=$1
  local name=$(basename "$cfg" .yaml)
  local ckpt="checkpoints/${name}/result.json"
  if [ -f "$ckpt" ]; then
    echo "[skip] $name (already done)"
    return
  fi
  if [[ "$name" == *_14B_* ]]; then
    GPUS=2
    export GRAD_CHECKPOINT=1
  else
    GPUS=1
    unset GRAD_CHECKPOINT || true
  fi
  echo "[run ] $name (GPUs=$GPUS)"
  bash scripts/run.sh --gpus "$GPUS" --config "$cfg" 2>&1 | tee -a "$LOG"
}

for cfg in $CONFIGS; do
  run_one "$cfg"
done

echo "Done. Log: $LOG"
