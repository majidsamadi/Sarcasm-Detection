#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
python src/select_final_model.py
