#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
decoder=gpt
train_args=()
while (( $# )); do
  case "$1" in
    --decoder)
      decoder="${2:?--decoder requires gpt or crossattn}"
      shift 2 ;;
    --decoder=*) decoder="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: bash scripts/warmup.sh --decoder gpt|crossattn [scripts/train.py options]"
      echo "GPT is shared by Task 3/4; crossattn is the Task 2 model reused by Task 5."
      exit 0 ;;
    *) train_args+=("$1"); shift ;;
  esac
done
case "$decoder" in
  gpt) task=scaffold_editing; conf=examples/editing.yaml ;;
  crossattn) task=translation; conf=examples/translation.yaml ;;
  *) echo "Unknown decoder: $decoder (expected gpt or crossattn)" >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" scripts/train.py --task "$task" --conf "$conf" \
  --phase warmup --init-checkpoint checkpoints/pretrain_stage2.ckpt \
  --log-dir "experiments/warmup_$decoder" "${train_args[@]}"
