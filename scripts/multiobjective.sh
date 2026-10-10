#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/finetune/task4_data_genenrate/generate.py --phase finetune "$@"
