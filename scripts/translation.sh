#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/train.py --task translation --conf examples/translation.yaml "$@"
