#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
# Sequential FG26 -> QM9S Stage II adaptation, not mixed-dataset training.
# Epoch/LR defaults follow the original QM9S run; override on the command line.
"$PYTHON_BIN" scripts/train.py --task pretrain --conf examples/pretrain.yaml \
  --dataset qm9s --log-dir experiments/pretrain_qm9s \
  --init-checkpoint "${STAGE2_CHECKPOINT:-checkpoints/pretrain_stage2.ckpt}" \
  --num-epochs 500 --lr 0.0003 --seed 1 \
  --reconstruct-weight 5 --denoising-weight 1 \
  --sp-struct-contrastive-weight 1 --sp-smiles-contrastive-weight 0.1 \
  --struct-smiles-contrastive-weight 0.1 --position-noise-scale 0.04 \
  --p-peak 0.5 --p-noise 0.1 \
  --spec-augment-strength 1.0 \
  --spec-aug-clean-prob 0.25 \
  --spec-aug-global-only-prob 0.30 \
  --spec-aug-global-intensity-prob 0.20 \
  --spec-aug-global-broadening-prob 0.10 \
  --spec-aug-global-local-shift-prob 0.15 \
  --spec-aug-uv-scale-min 0.94 --spec-aug-uv-scale-max 1.06 \
  --spec-aug-uv-shift-max 30 --spec-aug-uv-local-shift-max 25 \
  --spec-aug-uv-intensity-sigma 0.55 --spec-aug-uv-broadening-max 10 \
  --spec-aug-ir-raman-scale-min 0.97 --spec-aug-ir-raman-scale-max 1.03 \
  --spec-aug-ir-raman-shift-max 20 --spec-aug-ir-raman-local-shift-max 15 \
  --spec-aug-ir-raman-intensity-sigma 0.50 --spec-aug-ir-raman-broadening-max 4 \
  "$@"
