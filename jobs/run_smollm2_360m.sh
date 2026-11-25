#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" run_colab_exp_b.py --model smollm --gpu A100 "$@"
