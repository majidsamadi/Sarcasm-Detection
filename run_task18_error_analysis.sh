#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
python src/error_analysis.py "$@"
