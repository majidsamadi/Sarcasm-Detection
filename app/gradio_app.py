#!/usr/bin/env python3
"""Polished Gradio interface for the sarcasm detection NLP project.

This app makes Gradio the primary stakeholder-facing interface.
It focuses on clear presentation, readable cards, compact metrics, and a
complete interactive project walkthrough.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.predict_sarcasm import SarcasmPredictor
except Exception as exc:  # pragma: no cover
    SarcasmPredictor = None  # type: ignore
    PREDICTOR_IMPORT_ERROR = exc
else:
    PREDICTOR_IMPORT_ERROR = None

APP_TITLE = "Sarcasm Detection NLP Lab"
FINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.json"
UI_RUNS_DIR = PROJECT_ROOT / "reports" / "ui_runs"
UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)

TASKS: List[Dict[str, Any]] = [
    {"id": "T09", "title": "Preprocessing Version A", "stage": "Data", "runner": "run_task09_10_exact.sh", "heavy": True, "outputs": ["data/processed/A.csv"]},
    {"id": "T10", "title": "Preprocessing Version B", "stage": "Data", "runner": "run_task09_10_exact.sh", "heavy": True, "outputs": ["data/processed/B.csv"]},
    {"id": "T11", "title": "Train/Validation/Test Split", "stage": "Data", "runner": "run_task11_splits.sh", "heavy": False, "outputs": ["reports/task11_split_summary.md"]},
    {"id": "T12", "title": "Experiment Design", "stage": "Methodology", "runner": "run_task12_experiment_design.sh", "heavy": False, "outputs": ["configs/task12_experiment_design.json"]},
    {"id": "T13", "title": "Train BERTweet", "stage": "Training", "runner": "run_task13_train_bertweet.sh", "heavy": True, "outputs": ["reports/task13/E01_BERTweet_VersionA_metrics.json", "reports/task13/E02_BERTweet_VersionB_metrics.json"]},
    {"id": "T14", "title": "Train RoBERTa", "stage": "Training", "runner": "run_task14_train_roberta.sh", "heavy": True, "outputs": ["reports/task14/E03_RoBERTa_VersionA_metrics.json", "reports/task14/E04_RoBERTa_VersionB_metrics.json"]},
    {"id": "T15", "title": "Full Test Evaluation", "stage": "Evaluation", "runner": "run_task15_model_evaluation.sh", "heavy": True, "outputs": ["reports/task15/task15_model_evaluation_summary.md"]},
    {"id": "T16", "title": "Stopword Impact", "stage": "Evaluation", "runner": "run_task16_stopword_impact_analysis.sh", "heavy": False, "outputs": ["reports/task16/task16_stopword_impact_summary.md"]},
    {"id": "T17", "title": "Model Comparison", "stage": "Evaluation", "runner": "run_task17_model_comparison.sh", "heavy": False, "outputs": ["reports/task17/task17_model_comparison_summary.md"]},
    {"id": "T18", "title": "Error Analysis", "stage": "Interpretation", "runner": "run_task18_error_analysis.sh", "heavy": True, "outputs": ["reports/task18/task18_error_analysis_summary.md"]},
    {"id": "T19", "title": "Final Model Selection", "stage": "Finalization", "runner": "run_task19_final_model_selection.sh", "heavy": False, "outputs": ["configs/final_model_config.json", "reports/task19/final_model_card.md"]},
    {"id": "T20", "title": "Initial Demo", "stage": "Interface", "runner": "run_task20_demo.sh", "heavy": False, "outputs": ["src/predict_sarcasm.py"]},
    {"id": "T21", "title": "Ethics and Limitations", "stage": "Responsible AI", "runner": "run_task21_ethics_limitations.sh", "heavy": False, "outputs": ["reports/task21/task21_ethics_and_limitations_summary.md"]},
    {"id": "T22", "title": "Final Documentation", "stage": "Documentation", "runner": "run_task22_final_report.sh", "heavy": False, "outputs": ["reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md"]},
    {"id": "T23", "title": "Gradio Interface", "stage": "Interface", "runner": "run_gradio_dashboard.sh", "heavy": False, "outputs": ["app/gradio_app.py"]},
]


def rel(path: str | Path) -> Path:
    return PROJECT_ROOT / path


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def safe_read_text(path: str | Path, default: str = "") -> str:
    p = rel(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8", errors="replace")


def safe_read_json(path: str | Path) -> Dict[str, Any]:
    p = rel(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def nested_get(data: Dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        current: Any = data
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, "", {}):
            return current
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metric(data: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        value = nested_get(data, key, default=None)
        if value is not None:
            return as_float(value, default or 0.0)
    return default


def load_metrics_table() -> pd.DataFrame:
    paths = {
        "E01_BERTweet_VersionA": "reports/task15/E01_BERTweet_VersionA_test_metrics.json",
        "E02_BERTweet_VersionB": "reports/task15/E02_BERTweet_VersionB_test_metrics.json",
        "E03_RoBERTa_VersionA": "reports/task15/E03_RoBERTa_VersionA_test_metrics.json",
        "E04_RoBERTa_VersionB": "reports/task15/E04_RoBERTa_VersionB_test_metrics.json",
    }
    rows: List[Dict[str, Any]] = []
    for exp, path in paths.items():
        data = safe_read_json(path)
        if not data:
            continue
        version = "A" if "VersionA" in exp else "B"
        model = "BERTweet" if "BERTweet" in exp else "RoBERTa"
        rows.append(
            {
                "Experiment": exp,
                "Model": data.get("model_family", model),
                "Version": version,
                "Preprocessing": data.get("preprocessing", "Stopwords kept" if version == "A" else "Selective stopword removal"),
                "Accuracy": round(metric(data, "metrics.accuracy", "accuracy", default=0.0) or 0.0, 4),
                "Macro-F1": round(metric(data, "metrics.macro_f1", "macro_f1", default=0.0) or 0.0, 4),
                "Weighted-F1": round(metric(data, "metrics.weighted_f1", "weighted_f1", default=0.0) or 0.0, 4),
                "Macro Precision": round(metric(data, "metrics.macro_precision", "macro_precision", default=0.0) or 0.0, 4),
                "Macro Recall": round(metric(data, "metrics.macro_recall", "macro_recall", default=0.0) or 0.0, 4),
                "Rows": int(data.get("rows_evaluated", data.get("test_rows", 0)) or 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def compact_model_info() -> Dict[str, str]:
    final_config = safe_read_json("configs/final_model_config.json")
    final_summary = safe_read_json("reports/task19/task19_final_model_selection_summary.json")
    df = load_metrics_table()

    selected = nested_get(
        final_summary,
        "selected_experiment",
        "selected_model.experiment_id",
        "best_experiment.experiment_id",
        default=None,
    ) or nested_get(
        final_config,
        "selected_experiment",
        "selected_model.experiment_id",
        "final_model.experiment_id",
        "experiment_id",
        default="E03_RoBERTa_VersionA",
    )
    if isinstance(selected, dict):
        selected = selected.get("experiment_id") or selected.get("name") or "E03_RoBERTa_VersionA"
    selected = str(selected)

    model_family = nested_get(final_summary, "model_family", "selected_model.model_family", default=None) or nested_get(final_config, "model_family", "selected_model.model_family", "final_model.model_family", default="RoBERTa")
    version = "A" if "VersionA" in selected or "Version A" in selected else "B" if "VersionB" in selected or "Version B" in selected else "A"
    preprocessing = nested_get(final_config, "preprocessing", "selected_model.preprocessing", "final_model.preprocessing", default="Stopwords kept")

    accuracy = 0.7223
    macro_f1 = 0.7167
    weighted_f1 = 0.7165
    if not df.empty:
        top = df.iloc[0]
        accuracy = as_float(top.get("Accuracy"), accuracy)
        macro_f1 = as_float(top.get("Macro-F1"), macro_f1)
        weighted_f1 = as_float(top.get("Weighted-F1"), weighted_f1)
        model_family = str(top.get("Model", model_family))
        version = str(top.get("Version", version))
        preprocessing = str(top.get("Preprocessing", preprocessing))

    return {
        "display": f"{model_family} V{version}",
        "selected": selected,
        "model_family": str(model_family),
        "version": str(version),
        "preprocessing": str(preprocessing),
        "accuracy": f"{accuracy:.4f}",
        "macro_f1": f"{macro_f1:.4f}",
        "weighted_f1": f"{weighted_f1:.4f}",
    }


def task_status_df() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        outputs = task["outputs"]
        passed = sum(1 for output in outputs if rel(output).exists())
        done = passed == len(outputs) and len(outputs) > 0
        rows.append(
            {
                "Task": task["id"],
                "Stage": task["stage"],
                "Name": task["title"],
                "Status": "✅ Done" if done else "⚠️ Local file missing",
                "Checks": f"{passed}/{len(outputs)}",
                "Heavy task": "Yes" if task.get("heavy") else "No",
            }
        )
    return pd.DataFrame(rows)


def workflow_progress() -> Tuple[int, int, float]:
    df = task_status_df()
    completed = int(df["Status"].str.contains("Done", regex=False).sum()) if not df.empty else 0
    total = len(df)
    pct = round((completed / total) * 100, 1) if total else 0.0
    return completed, total, pct


def hero_html() -> str:
    info = compact_model_info()
    completed, total, pct = workflow_progress()
    return f"""
    <section class="hero-clean">
      <div class="hero-copy">
        <div class="kicker">WQF7007 NLP Project • Group 21</div>
        <h1>Sarcasm Detection<br><span>NLP Dashboard</span></h1>
        <p>A polished Gradio interface for exploring the complete sarcasm detection pipeline: preprocessing, model training, evaluation, model comparison, error analysis, ethics, and live prediction.</p>
        <div class="pill-row">
          <span>RoBERTa final model</span>
          <span>Context-aware input</span>
          <span>Stopwords kept</span>
          <span>Research demo</span>
        </div>
      </div>
      <div class="hero-metrics">
        <div class="hero-card"><small>Final model</small><strong>{esc(info['display'])}</strong><em>{esc(info['preprocessing'])}</em></div>
        <div class="hero-card"><small>Test accuracy</small><strong>{esc(info['accuracy'])}</strong><em>Held-out split</em></div>
        <div class="hero-card"><small>Test Macro-F1</small><strong>{esc(info['macro_f1'])}</strong><em>Primary metric</em></div>
        <div class="hero-card"><small>Workflow</small><strong>{completed}/{total}</strong><em>{pct:.1f}% complete</em></div>
      </div>
    </section>
    <div class="progress-track"><div style="width:{pct}%"></div></div>
    """


@lru_cache(maxsize=8)
def get_predictor(device: str = "auto"):
    if SarcasmPredictor is None:
        raise RuntimeError(f"Prediction module could not be imported: {PREDICTOR_IMPORT_ERROR}")
    return SarcasmPredictor(config_path=FINAL_CONFIG_PATH, device=device)


def empty_prediction_card() -> str:
    return """
    <div class="result-card waiting">
      <div class="result-badge">Waiting for input</div>
      <h2>Enter text to classify</h2>
      <p>Add a Reddit reply, optionally with parent-comment context, then click <b>Predict sarcasm</b>.</p>
      <div class="mini-grid">
        <span>Model: RoBERTa V-A</span>
        <span>Max length: 128</span>
      </div>
    </div>
    """


def predict(parent_comment: str, comment: str, device: str = "auto") -> Tuple[str, Dict[str, float], str, Dict[str, Any]]:
    try:
        predictor = get_predictor(device)
        result = predictor.predict(comment=comment, parent_comment=parent_comment)
        prob_non = float(result.probability_non_sarcastic)
        prob_sarc = float(result.probability_sarcastic)
        confidence = float(result.confidence)
        is_sarc = result.label.lower().startswith("sarcastic")
        status_class = "sarcastic" if is_sarc else "sincere"
        emoji = "😏" if is_sarc else "🙂"
        verdict = "Sarcastic" if is_sarc else "Non-sarcastic"
        short_note = "The model detected sarcastic intent." if is_sarc else "The model interpreted this as sincere or literal."

        html_card = f"""
        <div class="result-card {status_class}">
          <div class="result-badge">Prediction result</div>
          <h2>{emoji} {verdict}</h2>
          <p>{short_note}</p>
          <div class="confidence-line"><span>Confidence</span><b>{confidence:.4f}</b></div>
          <div class="bar-item"><div class="bar-label"><span>Non-sarcastic</span><b>{prob_non:.4f}</b></div><div class="bar"><i style="width:{prob_non*100:.1f}%"></i></div></div>
          <div class="bar-item"><div class="bar-label"><span>Sarcastic</span><b>{prob_sarc:.4f}</b></div><div class="bar alt"><i style="width:{prob_sarc*100:.1f}%"></i></div></div>
        </div>
        """
        safe_technical = {
            "label": verdict,
            "confidence": round(confidence, 4),
            "probability_non_sarcastic": round(prob_non, 4),
            "probability_sarcastic": round(prob_sarc, 4),
            "checkpoint_path": result.checkpoint_path,
            "model_name": result.model_name,
        }
        return html_card, {"Non-sarcastic": prob_non, "Sarcastic": prob_sarc}, result.combined_text, safe_technical
    except Exception as exc:
        return f"""
        <div class="result-card error">
          <div class="result-badge">Prediction failed</div>
          <h2>⚠️ Model unavailable</h2>
          <p>{esc(exc)}</p>
          <p class="tiny">Check that <code>models/roberta/versionA</code> exists locally or update the model path in <code>configs/final_model_config.json</code>.</p>
        </div>
        """, {}, "", {"error": str(exc)}


def model_metric_figure(metric_name: str = "Macro-F1"):
    df = load_metrics_table()
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    if df.empty or metric_name not in df.columns:
        ax.text(0.5, 0.5, "No metric files found", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig
    labels = [f"{row['Model']} V{row['Version']}" for _, row in df.iterrows()]
    values = df[metric_name].astype(float).tolist()
    colors = ["#16a34a" if i == 0 else "#7c3aed" if "RoBERTa" in labels[i] else "#0ea5e9" for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, alpha=0.88)
    ax.set_title(f"Model comparison by {metric_name}", fontsize=14, pad=14, weight="bold")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, max(1.0, max(values) + 0.08))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.4f}", ha="center", fontsize=10, weight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def stopword_figure():
    df = load_metrics_table()
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    if df.empty:
        ax.text(0.5, 0.5, "No metric files found", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig
    pivot = df.pivot_table(index="Model", columns="Version", values="Macro-F1", aggfunc="first")
    models = list(pivot.index)
    x = range(len(models))
    width = 0.34
    kept = [float(pivot.loc[m].get("A", 0.0)) for m in models]
    removed = [float(pivot.loc[m].get("B", 0.0)) for m in models]
    ax.bar([i - width / 2 for i in x], kept, width=width, label="Version A • kept", color="#16a34a", alpha=0.88)
    ax.bar([i + width / 2 for i in x], removed, width=width, label="Version B • removed", color="#ef4444", alpha=0.78)
    ax.set_title("Stopword impact on Macro-F1", fontsize=14, pad=14, weight="bold")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(1.0, max(kept + removed) + 0.08))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return fig


def confusion_figure(experiment: str = "E03_RoBERTa_VersionA"):
    data = safe_read_json(f"reports/task15/{experiment}_test_metrics.json")
    cm = nested_get(data, "confusion_matrix", "metrics.confusion_matrix", default=None)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    if not cm:
        ax.text(0.5, 0.5, "Confusion matrix not found", ha="center", va="center")
        ax.axis("off")
        return fig
    import numpy as np
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(experiment.replace("_", " "), fontsize=11, weight="bold")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:,}", ha="center", va="center", color="#111827", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def report_files() -> List[str]:
    candidates: List[str] = []
    for folder in ["reports", "docs"]:
        base = rel(folder)
        if not base.exists():
            continue
        for pattern in ["*.md", "*.txt", "*.json", "*.csv"]:
            candidates.extend(str(p.relative_to(PROJECT_ROOT)) for p in base.rglob(pattern))
    priority = [
        "reports/task15/task15_model_evaluation_summary.md",
        "reports/task17/task17_model_comparison_summary.md",
        "reports/task18/task18_error_analysis_summary.md",
        "reports/task19/final_model_card.md",
        "reports/task21/task21_ethics_and_limitations_summary.md",
    ]
    ordered = [p for p in priority if p in candidates] + sorted(p for p in candidates if p not in priority)
    return ordered[:250]


def load_report(path: str) -> Tuple[str, Optional[pd.DataFrame]]:
    if not path:
        return "Select a report first.", None
    p = rel(path)
    if not p.exists():
        return f"Report not found: `{path}`", None
    suffix = p.suffix.lower()
    if suffix == ".csv":
        try:
            return f"### {path}\n\nCSV preview below.", pd.read_csv(p).head(200)
        except Exception as exc:
            return f"Could not load CSV: {exc}", None
    if suffix == ".json":
        try:
            return "```json\n" + json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2)[:8000] + "\n```", None
        except Exception:
            return "```\n" + p.read_text(encoding="utf-8", errors="replace")[:8000] + "\n```", None
    return p.read_text(encoding="utf-8", errors="replace")[:12000], None


def runner_choices() -> List[str]:
    return [f"{task['id']} • {task['title']}" for task in TASKS if rel(task["runner"]).exists()]


def run_task(choice: str, confirm_heavy: bool, quick_mode: bool) -> str:
    allow = os.getenv("GRADIO_ALLOW_TASK_RUNS", "1").lower() in {"1", "true", "yes"}
    if not allow:
        return "Task execution is disabled for this environment. Set GRADIO_ALLOW_TASK_RUNS=1 for local use."
    if not choice:
        return "Select a task first."
    task_id = choice.split("•", 1)[0].strip()
    task = next((t for t in TASKS if t["id"] == task_id), None)
    if not task:
        return "Unknown task."
    if task.get("heavy") and not confirm_heavy:
        return "This is a heavy task. Tick the confirmation box before running it."
    runner = rel(task["runner"])
    if not runner.exists():
        return f"Runner not found: {task['runner']}"
    env = os.environ.copy()
    if quick_mode:
        env.update({
            "TASK13_MAX_TRAIN_SAMPLES": "5000",
            "TASK13_MAX_VALID_SAMPLES": "1000",
            "TASK14_MAX_TRAIN_SAMPLES": "5000",
            "TASK14_MAX_VALID_SAMPLES": "1000",
            "TASK15_MAX_TEST_SAMPLES": "5000",
        })
    log_name = f"{task_id.lower()}_{int(time.time())}.log"
    log_path = UI_RUNS_DIR / log_name
    process = subprocess.run(["bash", str(runner)], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60 * 60)
    combined = (process.stdout or "") + "\n" + (process.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    status = "completed" if process.returncode == 0 else f"failed with exit code {process.returncode}"
    return f"Task {task_id} {status}. Log saved to {log_path.relative_to(PROJECT_ROOT)}\n\n" + combined[-10000:]


def overview_markdown() -> str:
    info = compact_model_info()
    return f"""
### Project snapshot

This dashboard presents the complete sarcasm detection workflow in one place. The project compares **BERTweet** and **RoBERTa** under two preprocessing settings: Version A keeps stopwords, while Version B selectively removes stopwords but preserves important negations.

The final selected model is **{info['display']}**, using **{info['preprocessing']}**. It achieved **{info['accuracy']} accuracy** and **{info['macro_f1']} Macro-F1** on the held-out test split.

### What users can do here

1. Test live sarcasm predictions with optional Reddit parent-comment context.  
2. View the task-by-task workflow status.  
3. Compare all four model experiments.  
4. Browse generated reports and evaluation files.  
5. Review error analysis, ethics, and hosting readiness notes.  
    """


def ethics_and_hosting_markdown() -> str:
    ethics = safe_read_text("reports/task21/task21_ethics_and_limitations_summary.md", "Task 21 ethics report was not found locally.")
    hosting = """
### Hosting plan

For local presentation, this Gradio interface loads the final checkpoint from `models/roberta/versionA`. For Hugging Face Spaces, the recommended setup is to upload the final model checkpoint to a Hugging Face model repository and update `configs/final_model_config.json` to load the model from that hosted path.

For public hosting, keep task execution disabled by setting:

```bash
GRADIO_ALLOW_TASK_RUNS=0
```

This keeps the hosted app focused on inference, reports, and project explanation instead of long-running training tasks.
    """
    return ethics + "\n\n---\n\n" + hosting


def custom_css() -> str:
    return """
    :root {
      --ink:#0f172a;
      --muted:#64748b;
      --line:#e2e8f0;
      --violet:#7c3aed;
      --cyan:#06b6d4;
      --green:#16a34a;
      --rose:#e11d48;
      --panel:#ffffff;
      --soft:#f8fafc;
    }
    html, body, .gradio-container {
      background: linear-gradient(180deg,#f8fbff 0%, #ffffff 45%, #f8fafc 100%) !important;
      color: var(--ink) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }
    .gradio-container {
      max-width: 1240px !important;
      margin: 0 auto !important;
      padding: 22px !important;
    }
    footer, .footer, .built-with, .settings {
      display: none !important;
      visibility: hidden !important;
    }
    .hero-clean {
      display:grid;
      grid-template-columns: 1.08fr .92fr;
      gap: 26px;
      background:
        radial-gradient(circle at 14% 12%, rgba(255,255,255,.25), transparent 28%),
        linear-gradient(135deg,#15152e 0%, #312e81 40%, #0891b2 100%);
      color:#fff;
      border-radius: 30px;
      padding: 38px;
      box-shadow: 0 24px 70px rgba(15,23,42,.22);
      overflow:hidden;
      margin-bottom: 18px;
    }
    .hero-copy h1 {
      margin: 10px 0 16px;
      font-size: clamp(38px, 6vw, 72px);
      line-height: .95;
      letter-spacing: -.06em;
      color:#fff;
    }
    .hero-copy h1 span { color:#c4f1ff; }
    .hero-copy p {
      max-width: 760px;
      font-size: 17px;
      line-height: 1.7;
      color: rgba(255,255,255,.86);
      margin: 0 0 24px;
    }
    .kicker {
      font-weight: 800;
      letter-spacing: .16em;
      text-transform: uppercase;
      font-size: 12px;
      color:#a5f3fc;
    }
    .pill-row {
      display:flex;
      flex-wrap:wrap;
      gap:10px;
    }
    .pill-row span {
      background: rgba(255,255,255,.12);
      border: 1px solid rgba(255,255,255,.2);
      border-radius: 999px;
      padding: 9px 13px;
      font-size: 13px;
      color:#fff;
      backdrop-filter: blur(8px);
    }
    .hero-metrics {
      display:grid;
      grid-template-columns: repeat(2,minmax(0,1fr));
      gap:14px;
      align-content:center;
    }
    .hero-card {
      min-width:0;
      border-radius: 22px;
      padding: 18px;
      background: rgba(255,255,255,.12);
      border: 1px solid rgba(255,255,255,.22);
      backdrop-filter: blur(10px);
      overflow:hidden;
    }
    .hero-card small {
      display:block;
      text-transform:uppercase;
      letter-spacing:.12em;
      font-size: 11px;
      color:#dbeafe;
      margin-bottom: 8px;
      font-weight:800;
    }
    .hero-card strong {
      display:block;
      font-size: clamp(22px, 3vw, 33px);
      line-height:1.08;
      color:#fff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .hero-card em {
      display:block;
      margin-top:8px;
      font-style:normal;
      color:rgba(255,255,255,.76);
      font-size: 13px;
    }
    .progress-track {
      height: 10px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow:hidden;
      margin: 12px 0 22px;
    }
    .progress-track div {
      height:100%;
      border-radius:999px;
      background: linear-gradient(90deg,var(--violet),var(--cyan));
    }
    .gradio-container button.primary, button.primary {
      background: linear-gradient(135deg,var(--violet),var(--cyan)) !important;
      border:0 !important;
      color:white !important;
      font-weight:800 !important;
      border-radius: 14px !important;
      box-shadow: 0 12px 28px rgba(124,58,237,.22) !important;
    }
    .result-card {
      border-radius: 26px;
      padding: 28px;
      background: #fff;
      border: 1px solid var(--line);
      box-shadow: 0 18px 40px rgba(15,23,42,.08);
      min-height: 280px;
    }
    .result-card h2 {
      margin: 12px 0 8px;
      font-size: 34px;
      letter-spacing:-.03em;
      color: var(--ink);
    }
    .result-card p {
      color: var(--muted);
      line-height: 1.55;
    }
    .result-card.waiting { background: linear-gradient(135deg,#ffffff,#f8fafc); }
    .result-card.sarcastic { background: linear-gradient(135deg,#fff7ed,#ffffff 60%); border-color:#fed7aa; }
    .result-card.sincere { background: linear-gradient(135deg,#ecfeff,#ffffff 60%); border-color:#bae6fd; }
    .result-card.error { background: linear-gradient(135deg,#fff1f2,#ffffff 60%); border-color:#fecdd3; }
    .result-badge {
      display:inline-flex;
      background:#eef2ff;
      color:#4f46e5;
      border-radius:999px;
      padding:7px 11px;
      font-size:12px;
      text-transform:uppercase;
      letter-spacing:.1em;
      font-weight:900;
    }
    .confidence-line {
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding: 14px 0;
      margin: 14px 0;
      border-top:1px solid var(--line);
      border-bottom:1px solid var(--line);
    }
    .confidence-line span { color:var(--muted); font-weight:700; }
    .confidence-line b { font-size: 24px; color:var(--ink); }
    .bar-item { margin: 15px 0; }
    .bar-label { display:flex; justify-content:space-between; font-size:13px; margin-bottom:8px; color:var(--ink); }
    .bar { height: 12px; border-radius: 999px; overflow:hidden; background:#e2e8f0; }
    .bar i { display:block; height:100%; background: linear-gradient(90deg,var(--green),#22c55e); border-radius:999px; }
    .bar.alt i { background: linear-gradient(90deg,var(--violet),#f97316); }
    .mini-grid { display:grid; grid-template-columns: repeat(2,1fr); gap:10px; margin-top:18px; }
    .mini-grid span { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:14px; padding:10px; color:#475569; font-size:13px; }
    .tiny { font-size: 13px !important; }
    .section-note {
      padding: 18px 20px;
      border-radius: 20px;
      background: #fff;
      border: 1px solid var(--line);
      box-shadow: 0 12px 30px rgba(15,23,42,.05);
      margin-bottom: 16px;
    }
    .section-note h3 { margin-top:0; color: var(--ink); }
    .section-note p { color: var(--muted); line-height:1.65; }
    .tabs button, .tab-nav button {
      font-weight: 800 !important;
      border-radius: 14px !important;
    }
    .dataframe, table {
      font-size: 13px !important;
    }
    textarea, input, select {
      border-radius: 16px !important;
    }
    @media (max-width: 980px) {
      .hero-clean { grid-template-columns: 1fr; padding: 28px; }
      .hero-metrics { grid-template-columns: 1fr; }
      .hero-card strong { white-space: normal; }
    }
    """


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate"),
        css=custom_css(),
    ) as demo:
        gr.HTML(hero_html())

        with gr.Tabs():
            with gr.Tab("🏠 Overview"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(overview_markdown())
                    with gr.Column(scale=1):
                        info = compact_model_info()
                        gr.HTML(
                            f"""
                            <div class="section-note">
                              <h3>Final model</h3>
                              <p><b>{esc(info['display'])}</b><br>{esc(info['preprocessing'])}</p>
                              <p><b>Accuracy:</b> {esc(info['accuracy'])}<br><b>Macro-F1:</b> {esc(info['macro_f1'])}</p>
                            </div>
                            """
                        )
                gr.Dataframe(value=task_status_df(), label="Workflow status", interactive=False, wrap=True, max_height=460)

            with gr.Tab("🔮 Live Prediction"):
                gr.HTML('<div class="section-note"><h3>Try the final model</h3><p>Use optional parent-comment context when the reply depends on conversation history. The app returns a prediction, confidence score, and class probabilities.</p></div>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        parent = gr.Textbox(label="Optional parent comment / context", lines=4, placeholder="Example: The deadline moved to tomorrow.")
                        comment = gr.Textbox(label="Comment to classify", lines=5, placeholder="Example: Perfect, I love surprise deadlines.")
                        device = gr.Dropdown(["auto", "cpu", "mps", "cuda"], value="auto", label="Inference device")
                        with gr.Row():
                            predict_btn = gr.Button("Predict sarcasm", variant="primary")
                            clear_btn = gr.ClearButton([parent, comment])
                        gr.Examples(
                            examples=[
                                ["The deadline moved to tomorrow.", "Perfect, I love surprise deadlines.", "auto"],
                                ["How was the seminar?", "It was useful and I learned several new methods.", "auto"],
                                ["The app crashed again.", "Amazing, exactly what I wanted today.", "auto"],
                                ["Did you enjoy the meeting?", "Yes, it was productive and clear.", "auto"],
                            ],
                            inputs=[parent, comment, device],
                        )
                    with gr.Column(scale=1):
                        pred_html = gr.HTML(empty_prediction_card())
                        label_output = gr.Label(label="Class probabilities", num_top_classes=2)
                        with gr.Accordion("Combined model input", open=False):
                            combined_text = gr.Textbox(label="Text sent to model", lines=4, interactive=False)
                        with gr.Accordion("Technical output", open=False):
                            json_output = gr.JSON(label="Compact technical details")
                predict_btn.click(predict, inputs=[parent, comment, device], outputs=[pred_html, label_output, combined_text, json_output])

            with gr.Tab("📊 Results"):
                gr.HTML('<div class="section-note"><h3>Model comparison</h3><p>All four experiments were evaluated on the same held-out test split. Macro-F1 is treated as the main selection metric.</p></div>')
                gr.Dataframe(value=load_metrics_table(), label="Held-out test metrics", interactive=False, wrap=True, max_height=260)
                with gr.Row():
                    metric_selector = gr.Dropdown(["Macro-F1", "Accuracy", "Weighted-F1", "Macro Precision", "Macro Recall"], value="Macro-F1", label="Chart metric")
                    metric_plot = gr.Plot(value=model_metric_figure("Macro-F1"), label="Metric comparison")
                metric_selector.change(model_metric_figure, inputs=metric_selector, outputs=metric_plot)
                stopword_plot = gr.Plot(value=stopword_figure(), label="Stopword impact")
                with gr.Row():
                    cm_choice = gr.Dropdown(
                        ["E03_RoBERTa_VersionA", "E04_RoBERTa_VersionB", "E01_BERTweet_VersionA", "E02_BERTweet_VersionB"],
                        value="E03_RoBERTa_VersionA",
                        label="Confusion matrix",
                    )
                    cm_plot = gr.Plot(value=confusion_figure("E03_RoBERTa_VersionA"), label="Confusion matrix")
                cm_choice.change(confusion_figure, inputs=cm_choice, outputs=cm_plot)

            with gr.Tab("🧭 Workflow"):
                gr.HTML('<div class="section-note"><h3>End-to-end project flow</h3><p>This section shows how the project moved from data preprocessing to model comparison, error analysis, final selection, UI, and responsible-use reporting.</p></div>')
                refresh_status = gr.Button("Refresh workflow status", variant="primary")
                status_df = gr.Dataframe(value=task_status_df(), label="Task completion matrix", interactive=False, wrap=True, max_height=560)
                refresh_status.click(task_status_df, outputs=status_df)

            with gr.Tab("📚 Reports"):
                gr.HTML('<div class="section-note"><h3>Reports explorer</h3><p>Browse generated Markdown, JSON, CSV, and text reports without leaving the dashboard.</p></div>')
                report_dropdown = gr.Dropdown(choices=report_files(), label="Report file", value="reports/task15/task15_model_evaluation_summary.md" if rel("reports/task15/task15_model_evaluation_summary.md").exists() else None)
                with gr.Row():
                    load_report_btn = gr.Button("Open report", variant="primary")
                    refresh_reports_btn = gr.Button("Refresh file list")
                report_md = gr.Markdown("Select a report and click **Open report**.")
                report_table = gr.Dataframe(label="CSV preview", interactive=False, wrap=True, max_height=400)
                load_report_btn.click(load_report, inputs=report_dropdown, outputs=[report_md, report_table])
                refresh_reports_btn.click(lambda: gr.Dropdown(choices=report_files()), outputs=report_dropdown)

            with gr.Tab("⚙️ Run Tasks"):
                gr.HTML('<div class="section-note"><h3>Local pipeline runner</h3><p>This tab is for local use only. Heavy tasks such as model training can take time. Disable this tab for public Hugging Face hosting with <code>GRADIO_ALLOW_TASK_RUNS=0</code>.</p></div>')
                task_select = gr.Dropdown(choices=runner_choices(), label="Task runner", value=runner_choices()[0] if runner_choices() else None)
                confirm_heavy = gr.Checkbox(label="I understand this may run a heavy local task", value=False)
                quick_mode = gr.Checkbox(label="Quick mode for supported heavy tasks", value=True)
                run_btn = gr.Button("Run selected task", variant="primary")
                run_log = gr.Textbox(label="Task log", lines=18, max_lines=30)
                run_btn.click(run_task, inputs=[task_select, confirm_heavy, quick_mode], outputs=run_log)

            with gr.Tab("🛡️ Ethics & Hosting"):
                gr.Markdown(ethics_and_hosting_markdown())

    return demo


demo = build_app()


def main() -> None:
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    main()
