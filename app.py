#!/usr/bin/env python3
"""Hugging Face Spaces entry point for the Gradio sarcasm detection UI."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "gradio_app.py"), run_name="__main__")
