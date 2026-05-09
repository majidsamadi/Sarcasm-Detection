#!/usr/bin/env python3
"""Task 17: Model Comparison.

This script consolidates Task 15 test metrics for the four controlled experiments:
E01 BERTweet Version A, E02 BERTweet Version B, E03 RoBERTa Version A, and
E04 RoBERTa Version B.

It generates report-ready tables, figures, and an academic summary. It does not
train or evaluate models again.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "task17_model_comparison_config.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "task17"
FIGURE_DIR = REPORT_DIR / "figures"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_key(key: str) -> str:
    return key.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def recursive_numeric_search(data: Any, candidate_keys: Iterable[str]) -> Optional[float]:
    """Search nested dict/list for the first numeric value matching candidate keys."""
    candidate_norm = {normalize_key(k) for k in candidate_keys}

    if isinstance(data, dict):
        # Direct key search first
        for k, v in data.items():
            if normalize_key(str(k)) in candidate_norm and is_number(v):
                return float(v)
        # Then nested search
        for v in data.values():
            found = recursive_numeric_search(v, candidate_keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = recursive_numeric_search(item, candidate_keys)
            if found is not None:
                return found
    return None


def get_metric(metrics: Dict[str, Any], candidate_keys: Iterable[str]) -> Optional[float]:
    return recursive_numeric_search(metrics, candidate_keys)


def get_from_classification_report_dict(metrics: Dict[str, Any], section: str, metric: str) -> Optional[float]:
    """Handle sklearn-style nested classification report if present in JSON."""
    candidates = [
        metrics.get("classification_report"),
        metrics.get("classification_report_dict"),
        metrics.get("report"),
    ]
    for report in candidates:
        if isinstance(report, dict):
            section_obj = report.get(section) or report.get(section.replace("_", " "))
            if isinstance(section_obj, dict):
                val = section_obj.get(metric) or section_obj.get(metric.replace("_", "-"))
                if is_number(val):
                    return float(val)
    return None


def parse_text_classification_report(path: Path) -> Dict[str, Optional[float]]:
    """Parse sklearn's text classification report as fallback."""
    parsed: Dict[str, Optional[float]] = {
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": None,
    }
    if not path.exists():
        return parsed

    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        # Example: accuracy                           0.72     96509
        if parts[0] == "accuracy" and len(parts) >= 2:
            try:
                parsed["accuracy"] = float(parts[-2]) if len(parts) >= 3 else float(parts[-1])
            except ValueError:
                pass
        # Example: macro avg       0.72      0.72      0.72     96509
        if len(parts) >= 6 and parts[0] == "macro" and parts[1] == "avg":
            try:
                parsed["macro_precision"] = float(parts[2])
                parsed["macro_recall"] = float(parts[3])
                parsed["macro_f1"] = float(parts[4])
            except ValueError:
                pass
        # Example: weighted avg    0.72      0.72      0.72     96509
        if len(parts) >= 6 and parts[0] == "weighted" and parts[1] == "avg":
            try:
                parsed["weighted_precision"] = float(parts[2])
                parsed["weighted_recall"] = float(parts[3])
                parsed["weighted_f1"] = float(parts[4])
            except ValueError:
                pass
    return parsed


def metric_or_fallback(
    metrics: Dict[str, Any],
    report_fallback: Dict[str, Optional[float]],
    key: str,
    candidate_keys: Iterable[str],
    report_section: Optional[str] = None,
    report_metric: Optional[str] = None,
) -> Optional[float]:
    val = get_metric(metrics, candidate_keys)
    if val is not None:
        return val
    if report_section and report_metric:
        val = get_from_classification_report_dict(metrics, report_section, report_metric)
        if val is not None:
            return val
    return report_fallback.get(key)


def round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def load_experiment_row(exp: Dict[str, Any]) -> Dict[str, Any]:
    metrics_path = PROJECT_ROOT / exp["metrics_path"]
    report_path = PROJECT_ROOT / exp.get("classification_report_path", "")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = load_json(metrics_path)
    report_fallback = parse_text_classification_report(report_path)

    accuracy = metric_or_fallback(
        metrics,
        report_fallback,
        "accuracy",
        ["accuracy", "acc", "test_accuracy"],
    )
    macro_precision = metric_or_fallback(
        metrics,
        report_fallback,
        "macro_precision",
        ["macro_precision", "precision_macro", "macro_avg_precision", "test_macro_precision"],
        "macro avg",
        "precision",
    )
    macro_recall = metric_or_fallback(
        metrics,
        report_fallback,
        "macro_recall",
        ["macro_recall", "recall_macro", "macro_avg_recall", "test_macro_recall"],
        "macro avg",
        "recall",
    )
    macro_f1 = metric_or_fallback(
        metrics,
        report_fallback,
        "macro_f1",
        ["macro_f1", "macro_f1_score", "f1_macro", "macro_avg_f1_score", "test_macro_f1"],
        "macro avg",
        "f1-score",
    )
    weighted_f1 = metric_or_fallback(
        metrics,
        report_fallback,
        "weighted_f1",
        ["weighted_f1", "weighted_f1_score", "f1_weighted", "weighted_avg_f1_score", "test_weighted_f1"],
        "weighted avg",
        "f1-score",
    )

    rows_evaluated = get_metric(metrics, ["rows_evaluated", "num_rows", "n_rows", "test_rows", "samples", "support"])

    return {
        "experiment_id": exp["experiment_id"],
        "short_id": exp["short_id"],
        "model": exp["model"],
        "model_family": exp["model_family"],
        "dataset_version": exp["dataset_version"],
        "preprocessing": exp["preprocessing"],
        "accuracy": round_or_none(accuracy),
        "macro_precision": round_or_none(macro_precision),
        "macro_recall": round_or_none(macro_recall),
        "macro_f1": round_or_none(macro_f1),
        "weighted_f1": round_or_none(weighted_f1),
        "rows_evaluated": int(rows_evaluated) if rows_evaluated is not None else None,
        "metrics_path": exp["metrics_path"],
    }


def require_metric(rows: List[Dict[str, Any]], metric: str) -> None:
    missing = [row["experiment_id"] for row in rows if row.get(metric) is None]
    if missing:
        raise ValueError(f"Missing required metric '{metric}' for experiments: {missing}")


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(r["macro_f1"] if r["macro_f1"] is not None else -1),
            float(r["accuracy"] if r["accuracy"] is not None else -1),
        ),
        reverse=True,
    )


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Rank",
        "Experiment",
        "Model",
        "Version",
        "Preprocessing",
        "Accuracy",
        "Macro-F1",
        "Macro Precision",
        "Macro Recall",
        "Weighted-F1",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in enumerate(rows, start=1):
        values = [
            idx,
            row["experiment_id"],
            row["model"],
            row["dataset_version"],
            row["preprocessing"],
            fmt(row["accuracy"]),
            fmt(row["macro_f1"]),
            fmt(row["macro_precision"]),
            fmt(row["macro_recall"]),
            fmt(row["weighted_f1"]),
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def model_family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for model in sorted({r["model_family"] for r in rows}):
        model_rows = [r for r in rows if r["model_family"] == model]
        best = sort_rows(model_rows)[0]
        summary[model] = {
            "best_experiment": best["experiment_id"],
            "best_version": best["dataset_version"],
            "best_preprocessing": best["preprocessing"],
            "best_macro_f1": best["macro_f1"],
            "best_accuracy": best["accuracy"],
        }
    return summary


def preprocessing_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for model in sorted({r["model_family"] for r in rows}):
        a = next((r for r in rows if r["model_family"] == model and r["dataset_version"] == "Version A"), None)
        b = next((r for r in rows if r["model_family"] == model and r["dataset_version"] == "Version B"), None)
        if a and b:
            delta = None
            if a["macro_f1"] is not None and b["macro_f1"] is not None:
                delta = round(float(a["macro_f1"]) - float(b["macro_f1"]), 4)
            results.append(
                {
                    "model": model,
                    "version_a_macro_f1": a["macro_f1"],
                    "version_b_macro_f1": b["macro_f1"],
                    "delta_a_minus_b": delta,
                    "preferred_preprocessing": "Version A - stopwords kept" if delta is not None and delta >= 0 else "Version B - selective stopword removal",
                }
            )
    return results


def create_bar_chart(rows: List[Dict[str, Any]], metric: str, output_path: Path, title: str, ylabel: str) -> None:
    labels = [r["short_id"] + "\n" + r["model"] + "\n" + r["dataset_version"].replace(" ", "") for r in rows]
    values = [float(r[metric]) if r[metric] is not None else 0.0 for r in rows]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(1.0, max(values) + 0.05))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_accuracy_f1_chart(rows: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [r["short_id"] for r in rows]
    accuracy_values = [float(r["accuracy"]) if r["accuracy"] is not None else 0.0 for r in rows]
    f1_values = [float(r["macro_f1"]) if r["macro_f1"] is not None else 0.0 for r in rows]
    x = list(range(len(rows)))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], accuracy_values, width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], f1_values, width, label="Macro-F1")
    plt.xticks(x, labels)
    plt.title("Accuracy vs Macro-F1 by Experiment")
    plt.ylabel("Score")
    plt.ylim(0, max(1.0, max(accuracy_values + f1_values) + 0.05))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "rank",
        "experiment_id",
        "model",
        "dataset_version",
        "preprocessing",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "rows_evaluated",
        "metrics_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {k: row.get(k) for k in fieldnames if k != "rank"}
            out["rank"] = rank
            writer.writerow(out)


def main() -> None:
    print("Task 17: Model Comparison")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config: {CONFIG_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_json(CONFIG_PATH)
    rows = [load_experiment_row(exp) for exp in config["experiments"]]
    require_metric(rows, "macro_f1")
    require_metric(rows, "accuracy")

    ranked_rows = sort_rows(rows)
    best = ranked_rows[0]
    family_summary = model_family_summary(rows)
    prep_summary = preprocessing_summary(rows)

    summary = {
        "selection_metric": config["selection_metric"],
        "secondary_metric": config["secondary_metric"],
        "best_experiment": best,
        "ranked_results": ranked_rows,
        "model_family_summary": family_summary,
        "preprocessing_summary": prep_summary,
    }

    json_path = REPORT_DIR / "task17_model_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = REPORT_DIR / "task17_model_comparison_table.csv"
    write_csv(ranked_rows, csv_path)

    create_bar_chart(
        ranked_rows,
        "macro_f1",
        FIGURE_DIR / "task17_macro_f1_comparison.png",
        "Task 17: Test Macro-F1 Comparison",
        "Macro-F1",
    )
    create_bar_chart(
        ranked_rows,
        "accuracy",
        FIGURE_DIR / "task17_accuracy_comparison.png",
        "Task 17: Test Accuracy Comparison",
        "Accuracy",
    )
    create_accuracy_f1_chart(ranked_rows, FIGURE_DIR / "task17_accuracy_vs_macro_f1.png")

    prep_lines = []
    for item in prep_summary:
        delta = item["delta_a_minus_b"]
        if delta is None:
            delta_text = "N/A"
        elif delta >= 0:
            delta_text = f"Version A is higher by {delta:.4f} Macro-F1"
        else:
            delta_text = f"Version B is higher by {abs(delta):.4f} Macro-F1"
        prep_lines.append(
            f"- **{item['model']}**: Version A Macro-F1 = {fmt(item['version_a_macro_f1'])}, "
            f"Version B Macro-F1 = {fmt(item['version_b_macro_f1'])}. {delta_text}."
        )

    md = f"""# Task 17: Model Comparison

## Purpose

This task compares the four controlled experiments defined in Task 12 and evaluated in Task 15. The comparison uses the held-out test-set results, not training or validation results. The final model should be selected mainly using **test Macro-F1**, with accuracy and error analysis used as supporting evidence.

## Ranked Model Comparison

{markdown_table(ranked_rows)}

## Best Model

The best experiment based on test Macro-F1 is:

- **Experiment:** {best['experiment_id']}
- **Model:** {best['model']}
- **Dataset version:** {best['dataset_version']}
- **Preprocessing:** {best['preprocessing']}
- **Accuracy:** {fmt(best['accuracy'])}
- **Macro-F1:** {fmt(best['macro_f1'])}
- **Macro Precision:** {fmt(best['macro_precision'])}
- **Macro Recall:** {fmt(best['macro_recall'])}

## Model-Family Summary

"""
    for model, item in family_summary.items():
        md += (
            f"- **{model}** best run: {item['best_experiment']} "
            f"({item['best_version']}, {item['best_preprocessing']}) "
            f"with Macro-F1 = {fmt(item['best_macro_f1'])} and Accuracy = {fmt(item['best_accuracy'])}.\n"
        )

    md += """
## Stopword / Preprocessing Comparison

"""
    md += "\n".join(prep_lines)

    md += f"""

## Academic Interpretation

The comparison shows which model and preprocessing version performs best under the same experimental setup. Because all four experiments use the same split design, text-only input format, and maximum sequence length, the comparison is methodologically fair.

The current best-performing model is **{best['experiment_id']}**, selected mainly because it has the highest test Macro-F1 score. Macro-F1 is emphasized because it balances performance across both classes instead of focusing only on overall accuracy.

## Generated Outputs

- `reports/task17/task17_model_comparison_summary.md`
- `reports/task17/task17_model_comparison_summary.json`
- `reports/task17/task17_model_comparison_table.csv`
- `reports/task17/figures/task17_macro_f1_comparison.png`
- `reports/task17/figures/task17_accuracy_comparison.png`
- `reports/task17/figures/task17_accuracy_vs_macro_f1.png`
- `reports/task17/task17_progress_note.txt`

## Note for Next Task

Task 18 should perform error analysis on the best-performing model and compare false positives versus false negatives. This will help explain where the selected model still struggles with sarcasm detection.
"""

    md_path = REPORT_DIR / "task17_model_comparison_summary.md"
    md_path.write_text(md, encoding="utf-8")

    progress_note = (
        f"Model comparison completed for all four experiments using held-out test metrics. "
        f"The best model is {best['experiment_id']} with accuracy {fmt(best['accuracy'])} "
        f"and macro-F1 {fmt(best['macro_f1'])}. Results, ranked comparison table, JSON summary, "
        f"and comparison figures were saved under reports/task17."
    )
    (REPORT_DIR / "task17_progress_note.txt").write_text(progress_note + "\n", encoding="utf-8")

    print("\nRanked results:")
    for idx, row in enumerate(ranked_rows, start=1):
        print(
            f"{idx}. {row['experiment_id']} | accuracy={fmt(row['accuracy'])} | "
            f"macro_f1={fmt(row['macro_f1'])} | preprocessing={row['preprocessing']}"
        )
    print("\nBest experiment:", best["experiment_id"])
    print("Saved report:", md_path)
    print("Saved JSON summary:", json_path)
    print("Saved CSV table:", csv_path)
    print("Saved figures:", FIGURE_DIR)
    print("Saved progress note:", REPORT_DIR / "task17_progress_note.txt")
    print("\nDone. Task 17 model comparison completed successfully.")


if __name__ == "__main__":
    main()
