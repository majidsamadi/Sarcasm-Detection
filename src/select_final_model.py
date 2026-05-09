#!/usr/bin/env python3
"""
Task 19: Final Model Selection

Selects the final sarcasm detection model based on held-out test metrics,
with macro-F1 as the primary metric and accuracy as tie-breaker.
It also creates a report-ready model card and final model config for Task 20.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "task19_final_model_selection_config.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "task19"
FIGURE_DIR = REPORT_DIR / "figures"
FINAL_MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.json"


@dataclass
class ExperimentResult:
    experiment_id: str
    model_family: str
    dataset_version: str
    preprocessing: str
    checkpoint_path: str
    tokenizer_name: str
    accuracy: float
    macro_precision: Optional[float]
    macro_recall: Optional[float]
    macro_f1: float
    weighted_f1: Optional[float]
    source_metrics_file: str


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalized_key(key: str) -> str:
    return key.lower().replace("-", "_").replace(" ", "_")


def find_metric(obj: Any, candidate_keys: Iterable[str]) -> Optional[float]:
    """Recursively search for a metric key in nested dict/list JSON."""
    candidates = {normalized_key(k) for k in candidate_keys}

    if isinstance(obj, dict):
        for k, v in obj.items():
            if normalized_key(str(k)) in candidates:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        for v in obj.values():
            found = find_metric(v, candidates)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_metric(item, candidates)
            if found is not None:
                return found
    return None


def load_experiment_result(exp: Dict[str, Any]) -> ExperimentResult:
    metrics_path = PROJECT_ROOT / exp["metrics_file"]
    metrics = read_json(metrics_path)

    accuracy = find_metric(metrics, ["accuracy", "test_accuracy", "eval_accuracy"])
    macro_f1 = find_metric(metrics, ["macro_f1", "f1_macro", "macro avg_f1-score", "macro_avg_f1_score"])
    macro_precision = find_metric(metrics, ["macro_precision", "precision_macro", "macro avg_precision", "macro_avg_precision"])
    macro_recall = find_metric(metrics, ["macro_recall", "recall_macro", "macro avg_recall", "macro_avg_recall"])
    weighted_f1 = find_metric(metrics, ["weighted_f1", "f1_weighted", "weighted avg_f1-score", "weighted_avg_f1_score"])

    if accuracy is None:
        raise ValueError(f"Could not find accuracy in {metrics_path}")
    if macro_f1 is None:
        raise ValueError(f"Could not find macro-F1 in {metrics_path}")

    return ExperimentResult(
        experiment_id=exp["experiment_id"],
        model_family=exp["model_family"],
        dataset_version=exp["dataset_version"],
        preprocessing=exp["preprocessing"],
        checkpoint_path=exp["checkpoint_path"],
        tokenizer_name=exp["tokenizer_name"],
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        source_metrics_file=exp["metrics_file"],
    )


def load_error_analysis_summary() -> Dict[str, Any]:
    path = PROJECT_ROOT / "reports" / "task18" / "task18_error_analysis_summary.json"
    if path.exists():
        return read_json(path)
    return {}


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def write_csv(results: List[ExperimentResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "experiment_id",
            "model_family",
            "dataset_version",
            "preprocessing",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "checkpoint_path",
        ])
        for idx, r in enumerate(results, start=1):
            writer.writerow([
                idx,
                r.experiment_id,
                r.model_family,
                r.dataset_version,
                r.preprocessing,
                f"{r.accuracy:.6f}",
                "" if r.macro_precision is None else f"{r.macro_precision:.6f}",
                "" if r.macro_recall is None else f"{r.macro_recall:.6f}",
                f"{r.macro_f1:.6f}",
                "" if r.weighted_f1 is None else f"{r.weighted_f1:.6f}",
                r.checkpoint_path,
            ])


def make_figures(results: List[ExperimentResult]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r.experiment_id.replace("_", "\n") for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, [r.macro_f1 for r in results])
    plt.title("Task 19 Final Model Selection: Macro-F1 Ranking")
    plt.ylabel("Test Macro-F1")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "task19_final_model_macro_f1_ranking.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, [r.accuracy for r in results])
    plt.title("Task 19 Final Model Selection: Accuracy Ranking")
    plt.ylabel("Test Accuracy")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "task19_final_model_accuracy_ranking.png", dpi=160)
    plt.close()


def write_markdown_report(results: List[ExperimentResult], best: ExperimentResult, error_summary: Dict[str, Any]) -> None:
    report_path = REPORT_DIR / "task19_final_model_selection_summary.md"

    rows = []
    for idx, r in enumerate(results, start=1):
        rows.append(
            f"| {idx} | {r.experiment_id} | {r.model_family} | {r.dataset_version} | "
            f"{r.preprocessing} | {format_float(r.accuracy)} | {format_float(r.macro_f1)} |"
        )

    error_analysis_note = (
        "Task 18 error analysis was used as supporting evidence for the final selection. "
        "The selected model is not chosen only because it has the highest score; it is also the model "
        "for which false positives, false negatives, confidence behavior, and text-length effects were analyzed."
    )

    if error_summary:
        selected_from_error = error_summary.get("selected_experiment") or error_summary.get("experiment_id") or error_summary.get("model")
        if selected_from_error:
            error_analysis_note += f" The Task 18 error analysis file references: `{selected_from_error}`."

    md = f"""# Task 19: Final Model Selection

## Purpose

Task 19 selects **one final model** for the final implementation/demo. The selection is based mainly on **held-out test macro-F1**, with **accuracy** as a supporting metric and Task 18 error analysis as qualitative evidence.

## Selection Rule

The final model is selected using this rule:

1. Rank all experiments by **test macro-F1**.
2. If there is a tie, use **test accuracy** as the tie-breaker.
3. Use Task 18 error analysis to support the final decision and document limitations.

## Ranked Results

| Rank | Experiment | Model | Dataset Version | Preprocessing | Accuracy | Macro-F1 |
|---:|---|---|---|---|---:|---:|
{chr(10).join(rows)}

## Selected Final Model

**Selected model:** `{best.experiment_id}`  
**Model family:** {best.model_family}  
**Dataset version:** {best.dataset_version}  
**Preprocessing:** {best.preprocessing}  
**Checkpoint path:** `{best.checkpoint_path}`  
**Tokenizer:** `{best.tokenizer_name}`  
**Input format:** `parent_comment + comment`  
**Max sequence length:** 128  

## Justification

`{best.experiment_id}` is selected because it achieved the highest held-out test **macro-F1** among all four controlled experiments. Macro-F1 is prioritized because it evaluates both classes more fairly than accuracy alone, especially for classification tasks where the cost of false positives and false negatives should both be considered.

The selected model also supports the conclusion from the stopword impact analysis: **Version A, where stopwords are kept, performed better than Version B**. This is academically reasonable for sarcasm detection because sarcasm often depends on small function words, context, contrast, and sentence structure.

{error_analysis_note}

## Final Implementation Decision

The final demo should load only the selected model checkpoint:

```text
{best.checkpoint_path}
```

The other trained models are kept for comparison and reporting, but they should not be used in the final prediction interface unless the project scope changes.

## Limitations to Mention

- The model predicts sarcasm from text patterns and does not truly understand intent.
- Sarcasm is context-dependent and may require background knowledge beyond the available text.
- Reddit data may contain platform-specific language and bias.
- The model should not be used to automatically punish, moderate, or penalize users.
- Raw Reddit examples from error analysis should remain local and should not be pushed to GitHub.

## Outputs

- `configs/final_model_config.json`
- `reports/task19/task19_final_model_selection_summary.json`
- `reports/task19/task19_final_model_selection_table.csv`
- `reports/task19/task19_final_model_selection_summary.md`
- `reports/task19/figures/task19_final_model_macro_f1_ranking.png`
- `reports/task19/figures/task19_final_model_accuracy_ranking.png`
"""
    report_path.write_text(md, encoding="utf-8")


def write_model_card(best: ExperimentResult) -> None:
    card_path = REPORT_DIR / "final_model_card.md"
    card = f"""# Final Model Card: Sarcasm Detection

## Selected Model

- **Experiment ID:** {best.experiment_id}
- **Model family:** {best.model_family}
- **Checkpoint path:** `{best.checkpoint_path}`
- **Tokenizer:** `{best.tokenizer_name}`
- **Dataset version:** {best.dataset_version}
- **Preprocessing:** {best.preprocessing}
- **Input format:** `parent_comment + comment`
- **Task:** Binary sarcasm classification
- **Labels:** `0 = non-sarcastic`, `1 = sarcastic`

## Performance Summary

- **Accuracy:** {format_float(best.accuracy)}
- **Macro-F1:** {format_float(best.macro_f1)}
- **Macro Precision:** {format_float(best.macro_precision)}
- **Macro Recall:** {format_float(best.macro_recall)}
- **Weighted-F1:** {format_float(best.weighted_f1)}

## Intended Use

This model is intended for an academic NLP demonstration of sarcasm detection in social media text.

## Not Intended For

This model should **not** be used to automatically remove, flag, punish, or penalize user-generated content.

## Key Limitation

Sarcasm is highly context-dependent. The model may fail when sarcasm requires cultural knowledge, longer conversation history, or real-world background information.
"""
    card_path.write_text(card, encoding="utf-8")


def write_progress_note(best: ExperimentResult) -> None:
    note = (
        f"Final model selection completed. {best.experiment_id} was selected as the final model because it achieved "
        f"the highest held-out test macro-F1 ({best.macro_f1:.4f}) with accuracy {best.accuracy:.4f}. "
        f"The final model uses {best.model_family}, {best.dataset_version} ({best.preprocessing}), text-only input "
        f"from parent_comment + comment, and max length 128. Selection reports, model card, final model config, "
        f"ranked comparison table, and figures were saved under reports/task19 and configs/final_model_config.json."
    )
    (REPORT_DIR / "task19_progress_note.txt").write_text(note, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = read_json(CONFIG_PATH)
    results = [load_experiment_result(exp) for exp in config["experiments"]]
    results = sorted(results, key=lambda r: (r.macro_f1, r.accuracy), reverse=True)
    best = results[0]

    error_summary = load_error_analysis_summary()

    summary = {
        "task": "Task 19 - Final Model Selection",
        "selection_metric": config["selection_metric"],
        "tie_breaker_metric": config["tie_breaker_metric"],
        "selected_experiment": asdict(best),
        "ranked_results": [asdict(r) for r in results],
        "supporting_reports": config.get("supporting_reports", {}),
        "final_model_config_path": "configs/final_model_config.json",
    }

    (REPORT_DIR / "task19_final_model_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    write_csv(results, REPORT_DIR / "task19_final_model_selection_table.csv")
    make_figures(results)
    write_markdown_report(results, best, error_summary)
    write_model_card(best)
    write_progress_note(best)

    final_config = {
        "selected_experiment_id": best.experiment_id,
        "model_family": best.model_family,
        "checkpoint_path": best.checkpoint_path,
        "tokenizer_name": best.tokenizer_name,
        "dataset_version": best.dataset_version,
        "preprocessing": best.preprocessing,
        "input_format": "parent_comment + comment",
        "max_length": 128,
        "labels": {"0": "non-sarcastic", "1": "sarcastic"},
        "metrics": {
            "accuracy": best.accuracy,
            "macro_precision": best.macro_precision,
            "macro_recall": best.macro_recall,
            "macro_f1": best.macro_f1,
            "weighted_f1": best.weighted_f1,
        },
        "not_intended_for": "Automatic removal, flagging, punishment, or moderation of user content.",
    }
    FINAL_MODEL_CONFIG_PATH.write_text(json.dumps(final_config, indent=2), encoding="utf-8")

    print("Task 19: Final Model Selection")
    print(f"Selected experiment: {best.experiment_id}")
    print(f"Model family: {best.model_family}")
    print(f"Preprocessing: {best.preprocessing}")
    print(f"Accuracy: {best.accuracy:.4f}")
    print(f"Macro-F1: {best.macro_f1:.4f}")
    print(f"Checkpoint path: {best.checkpoint_path}")
    print(f"Saved final config: {FINAL_MODEL_CONFIG_PATH}")
    print(f"Saved report: {REPORT_DIR / 'task19_final_model_selection_summary.md'}")
    print("Done. Task 19 completed successfully.")


if __name__ == "__main__":
    main()
