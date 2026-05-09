#!/usr/bin/env python3
"""Prediction utilities for the final sarcasm detection demo.

Task 20 uses the selected final model from Task 19:
RoBERTa Version A, where stopwords are kept.

The model is loaded from the local checkpoint by default:
models/roberta/versionA
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "roberta" / "versionA"
DEFAULT_BASE_MODEL = "roberta-base"


LABEL_MAP = {
    0: "Non-sarcastic",
    1: "Sarcastic",
}


@dataclass
class PredictionResult:
    label_id: int
    label: str
    confidence: float
    probability_non_sarcastic: float
    probability_sarcastic: float
    combined_text: str
    checkpoint_path: str
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_id": self.label_id,
            "label": self.label,
            "confidence": self.confidence,
            "probability_non_sarcastic": self.probability_non_sarcastic,
            "probability_sarcastic": self.probability_sarcastic,
            "combined_text": self.combined_text,
            "checkpoint_path": self.checkpoint_path,
            "model_name": self.model_name,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _nested_get(data: Dict[str, Any], *paths: str, default: Any = None) -> Any:
    """Read likely key paths from different config formats."""
    for path in paths:
        current: Any = data
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return default


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def combine_text(parent_comment: str = "", comment: str = "") -> str:
    """Match the text-only input strategy used in training.

    Training used Reddit context by combining parent_comment + comment.
    The parent comment is optional for demo use.
    """
    parent = normalize_whitespace(parent_comment)
    child = normalize_whitespace(comment)
    if parent and child:
        return f"{parent} {child}"
    return child or parent


def select_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class SarcasmPredictor:
    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        checkpoint_path: Optional[Path | str] = None,
        device: str = "auto",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = _load_json(self.config_path)

        configured_checkpoint = _nested_get(
            self.config,
            "checkpoint_path",
            "selected_model.checkpoint_path",
            "final_model.checkpoint_path",
            "model.checkpoint_path",
            default=str(DEFAULT_CHECKPOINT_PATH),
        )
        self.checkpoint_path = Path(checkpoint_path or configured_checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = PROJECT_ROOT / self.checkpoint_path

        self.model_name = _nested_get(
            self.config,
            "model_name",
            "base_model",
            "base_model_name",
            "selected_model.base_model",
            "selected_model.model_name",
            "final_model.base_model",
            "final_model.model_name",
            default=DEFAULT_BASE_MODEL,
        )

        self.max_length = int(
            _nested_get(
                self.config,
                "max_length",
                "training.max_length",
                "inference.max_length",
                "selected_model.max_length",
                "final_model.max_length",
                default=128,
            )
        )

        self.device = select_device(device)
        self.tokenizer = self._load_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_path))
        self.model.to(self.device)
        self.model.eval()

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(str(self.checkpoint_path), use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    @torch.inference_mode()
    def predict(self, comment: str, parent_comment: str = "") -> PredictionResult:
        combined = combine_text(parent_comment=parent_comment, comment=comment)
        if not combined:
            raise ValueError("Please provide a comment or parent/comment text.")

        encoded = self.tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        prob_non = float(probs[0].item())
        prob_sarc = float(probs[1].item())
        label_id = int(torch.argmax(probs).item())
        confidence = max(prob_non, prob_sarc)

        return PredictionResult(
            label_id=label_id,
            label=LABEL_MAP.get(label_id, str(label_id)),
            confidence=confidence,
            probability_non_sarcastic=prob_non,
            probability_sarcastic=prob_sarc,
            combined_text=combined,
            checkpoint_path=str(self.checkpoint_path.relative_to(PROJECT_ROOT) if self.checkpoint_path.is_relative_to(PROJECT_ROOT) else self.checkpoint_path),
            model_name=str(self.model_name),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local sarcasm prediction using the selected final model.")
    parser.add_argument("--comment", required=True, help="The Reddit comment/reply text to classify.")
    parser.add_argument("--parent", default="", help="Optional parent comment context.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Inference device.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args()

    predictor = SarcasmPredictor(device=args.device)
    result = predictor.predict(comment=args.comment, parent_comment=args.parent)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Prediction: {result.label}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"P(non-sarcastic): {result.probability_non_sarcastic:.4f}")
        print(f"P(sarcastic): {result.probability_sarcastic:.4f}")
        print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
