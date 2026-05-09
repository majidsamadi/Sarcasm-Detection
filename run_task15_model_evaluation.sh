#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python src/evaluate_models.py "$@"
