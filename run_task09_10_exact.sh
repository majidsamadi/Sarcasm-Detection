#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  python3 -m venv .venv
  source .venv/bin/activate
fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/reproduce_task09_10_exact.py
