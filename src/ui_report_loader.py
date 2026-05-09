#!/usr/bin/env python3
"""Report loading utilities for the enhanced Streamlit dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskDefinition:
    number: int
    title: str
    category: str
    progress_file: Optional[str]
    primary_report: Optional[str]
    runner_sh: Optional[str]
    runner_bat: Optional[str]
    expected_outputs: List[str]
    heavy: bool = False
    local_only: bool = False

    @property
    def key(self) -> str:
        return f"task{self.number:02d}"


TASKS: List[TaskDefinition] = [
    TaskDefinition(9, "Preprocessing Version A", "Data Preparation", None, "docs/task09_10_exact_reproduction_note.md", "run_task09_10_exact.sh", None, ["data/processed/A.csv"], heavy=True, local_only=True),
    TaskDefinition(10, "Preprocessing Version B", "Data Preparation", None, "docs/task09_10_exact_reproduction_note.md", "run_task09_10_exact.sh", None, ["data/processed/B.csv"], heavy=True, local_only=True),
    TaskDefinition(11, "Train/Validation/Test Split", "Data Preparation", "reports/task11_progress_note.txt", "reports/task11_split_summary.md", "run_task11_splits.sh", "run_task11_splits.bat", ["reports/task11_split_summary.md"]),
    TaskDefinition(12, "Model Experiment Design", "Methodology", "reports/task12_progress_note.txt", "reports/task12_experiment_design_summary.md", "run_task12_experiment_design.sh", "run_task12_experiment_design.bat", ["configs/task12_experiment_design.json", "reports/task12_experiment_design_summary.md"]),
    TaskDefinition(13, "Train BERTweet", "Model Training", "reports/task13/task13_progress_note.txt", "reports/task13/task13_bertweet_training_summary.md", "run_task13_train_bertweet.sh", "run_task13_train_bertweet.bat", ["reports/task13/E01_BERTweet_VersionA_metrics.json", "reports/task13/E02_BERTweet_VersionB_metrics.json"], heavy=True, local_only=True),
    TaskDefinition(14, "Train RoBERTa", "Model Training", "reports/task14/task14_progress_note.txt", "reports/task14/task14_roberta_training_summary.md", "run_task14_train_roberta.sh", "run_task14_train_roberta.bat", ["reports/task14/E03_RoBERTa_VersionA_metrics.json", "reports/task14/E04_RoBERTa_VersionB_metrics.json"], heavy=True, local_only=True),
    TaskDefinition(15, "Model Evaluation", "Evaluation", "reports/task15/task15_progress_note.txt", "reports/task15/task15_model_evaluation_summary.md", "run_task15_model_evaluation.sh", "run_task15_model_evaluation.bat", ["reports/task15/E01_BERTweet_VersionA_test_metrics.json", "reports/task15/E02_BERTweet_VersionB_test_metrics.json", "reports/task15/E03_RoBERTa_VersionA_test_metrics.json", "reports/task15/E04_RoBERTa_VersionB_test_metrics.json"], heavy=True, local_only=True),
    TaskDefinition(16, "Stopword Impact Analysis", "Analysis", "reports/task16/task16_progress_note.txt", "reports/task16/task16_stopword_impact_summary.md", "run_task16_stopword_impact_analysis.sh", "run_task16_stopword_impact_analysis.bat", ["reports/task16/task16_stopword_impact_summary.json", "reports/task16/task16_stopword_impact_summary.md"]),
    TaskDefinition(17, "Model Comparison", "Analysis", "reports/task17/task17_progress_note.txt", "reports/task17/task17_model_comparison_summary.md", "run_task17_model_comparison.sh", "run_task17_model_comparison.bat", ["reports/task17/task17_model_comparison_summary.json", "reports/task17/task17_model_comparison_table.csv"]),
    TaskDefinition(18, "Error Analysis", "Analysis", "reports/task18/task18_progress_note.txt", "reports/task18/task18_error_analysis_summary.md", "run_task18_error_analysis.sh", "run_task18_error_analysis.bat", ["reports/task18/task18_error_analysis_summary.json", "reports/task18/task18_classification_report.txt"], heavy=True, local_only=True),
    TaskDefinition(19, "Final Model Selection", "Finalization", "reports/task19/task19_progress_note.txt", "reports/task19/task19_final_model_selection_summary.md", "run_task19_final_model_selection.sh", "run_task19_final_model_selection.bat", ["configs/final_model_config.json", "reports/task19/final_model_card.md"]),
    TaskDefinition(20, "Demo Interface", "Finalization", "reports/task20/task20_progress_note.txt", "reports/task20/task20_demo_summary.md", "run_task20_demo.sh", "run_task20_demo.bat", ["app/streamlit_app.py", "src/predict_sarcasm.py"]),
]


def project_path(relative: str | Path) -> Path:
    return PROJECT_ROOT / relative


def exists(relative: str | Path) -> bool:
    return project_path(relative).exists()


def read_text(relative: str | Path, default: str = "") -> str:
    path = project_path(relative)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(relative: str | Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = project_path(relative)
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default or {}


def load_csv(relative: str | Path) -> pd.DataFrame:
    return pd.read_csv(project_path(relative))


def task_completion(task: TaskDefinition) -> Dict[str, Any]:
    checks = [exists(p) for p in task.expected_outputs]
    progress_exists = exists(task.progress_file) if task.progress_file else False
    report_exists = exists(task.primary_report) if task.primary_report else False
    completed = all(checks) and (progress_exists or report_exists or len(task.expected_outputs) > 0)
    return {
        "task": task.key,
        "number": task.number,
        "title": task.title,
        "category": task.category,
        "completed": completed,
        "checks_passed": sum(checks),
        "checks_total": len(checks),
        "progress_exists": progress_exists,
        "report_exists": report_exists,
        "heavy": task.heavy,
        "local_only": task.local_only,
    }


def all_task_statuses() -> pd.DataFrame:
    return pd.DataFrame([task_completion(t) for t in TASKS])


def overall_progress() -> Dict[str, Any]:
    df = all_task_statuses()
    completed = int(df["completed"].sum()) if not df.empty else 0
    total = len(df)
    pct = completed / total if total else 0
    return {"completed": completed, "total": total, "pct": pct}


def metric_from_json(data: Dict[str, Any], keys: Iterable[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        current: Any = data
        ok = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok:
            try:
                return float(current)
            except Exception:
                continue
    return default


def load_task15_metrics() -> pd.DataFrame:
    paths = {
        "E01_BERTweet_VersionA": "reports/task15/E01_BERTweet_VersionA_test_metrics.json",
        "E02_BERTweet_VersionB": "reports/task15/E02_BERTweet_VersionB_test_metrics.json",
        "E03_RoBERTa_VersionA": "reports/task15/E03_RoBERTa_VersionA_test_metrics.json",
        "E04_RoBERTa_VersionB": "reports/task15/E04_RoBERTa_VersionB_test_metrics.json",
    }
    rows: List[Dict[str, Any]] = []
    for experiment, rel in paths.items():
        data = read_json(rel)
        if not data:
            continue
        model = "BERTweet" if "BERTweet" in experiment else "RoBERTa"
        version = "Version A" if "VersionA" in experiment else "Version B"
        preprocessing = "Stopwords kept" if "VersionA" in experiment else "Selective stopword removal"
        rows.append({
            "experiment": experiment,
            "model": model,
            "version": version,
            "preprocessing": preprocessing,
            "accuracy": metric_from_json(data, ["accuracy", "metrics.accuracy", "test_accuracy"]),
            "macro_f1": metric_from_json(data, ["macro_f1", "f1_macro", "metrics.macro_f1", "test_macro_f1"]),
            "weighted_f1": metric_from_json(data, ["weighted_f1", "f1_weighted", "metrics.weighted_f1", "test_weighted_f1"]),
            "macro_precision": metric_from_json(data, ["macro_precision", "precision_macro", "metrics.macro_precision"]),
            "macro_recall": metric_from_json(data, ["macro_recall", "recall_macro", "metrics.macro_recall"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "macro_f1" in df:
        df = df.sort_values("macro_f1", ascending=False, na_position="last")
    return df


def load_task17_table() -> pd.DataFrame:
    rel = "reports/task17/task17_model_comparison_table.csv"
    if exists(rel):
        try:
            return load_csv(rel)
        except Exception:
            pass
    return load_task15_metrics()


def load_final_model_summary() -> Dict[str, Any]:
    return read_json("reports/task19/task19_final_model_selection_summary.json")


def find_report_files() -> List[Dict[str, str]]:
    report_dirs = [PROJECT_ROOT / "reports", PROJECT_ROOT / "docs"]
    rows: List[Dict[str, str]] = []
    for base in report_dirs:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".csv"}:
                if "local_only" in path.parts or "error_samples" in path.parts or "ui_runs" in path.parts:
                    continue
                rel = str(path.relative_to(PROJECT_ROOT))
                rows.append({"file": rel, "type": path.suffix.lower(), "name": path.name})
    return rows


def extract_confusion_matrix(data: Dict[str, Any]) -> Optional[List[List[int]]]:
    for key in ["confusion_matrix", "metrics.confusion_matrix", "test_confusion_matrix"]:
        current: Any = data
        ok = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and isinstance(current, list):
            return current
    return None


def simple_markdown_preview(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n---\n_Preview truncated in UI._"


def parse_first_number(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None
