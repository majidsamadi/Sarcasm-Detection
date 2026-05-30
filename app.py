#!/usr/bin/env python3
"""Hugging Face Spaces entry point for the Gradio sarcasm detection UI."""
from pathlib import Path
import os
import runpy

# Hosted spaces should not run local training scripts by default.
os.environ.setdefault("GRADIO_ALLOW_TASK_RUNS", "0")
runpy.run_path(str(Path(__file__).resolve().parent / "app" / "gradio_app.py"), run_name="__main__")
