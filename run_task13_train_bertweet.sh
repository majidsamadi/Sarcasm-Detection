#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python src/train_bertweet.py --experiment all --overwrite
