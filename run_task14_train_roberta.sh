#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ARGS=()

if [ "${TASK14_FULL_DATA:-0}" = "1" ]; then
  ARGS+=(--full-data)
fi

if [ "${TASK14_OVERWRITE:-0}" = "1" ]; then
  ARGS+=(--overwrite)
fi

if [ -n "${TASK14_DEVICE:-}" ]; then
  ARGS+=(--device "${TASK14_DEVICE}")
fi

if [ -n "${TASK14_EXPERIMENT:-}" ]; then
  ARGS+=(--experiment "${TASK14_EXPERIMENT}")
fi

if [ -n "${TASK14_MAX_TRAIN_SAMPLES:-}" ]; then
  ARGS+=(--max-train-samples "${TASK14_MAX_TRAIN_SAMPLES}")
fi

if [ -n "${TASK14_MAX_VALID_SAMPLES:-}" ]; then
  ARGS+=(--max-valid-samples "${TASK14_MAX_VALID_SAMPLES}")
fi

python src/train_roberta.py "${ARGS[@]}"
