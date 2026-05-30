#!/usr/bin/env python3
"""Gradio interface for the sarcasm detection NLP project.

This Gradio app replaces the Streamlit interface as the primary stakeholder-facing UI.
It provides:
- live sarcasm prediction using the selected RoBERTa Version A model,
- full project workflow overview,
- model comparison dashboard,
- report explorer,
- optional local task runner,
- ethics and hosting-readiness notes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
FINAL_MODEL_NAME = "RoBERTa Version A"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "roberta" / "versionA"
FINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.json"
UI_RUNS_DIR = PROJECT_ROOT / "reports" / "ui_runs"
UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_COLOR = "#7c3aed"
SECONDARY_COLOR = "#06b6d4"
DARK_COLOR = "#111827"
CARD_BG = "rgba(255,255,255,0.92)"

TASKS: List[Dict[str, Any]] = [
    {"id": "T09", "title": "Preprocessing Version A", "category": "Data", "runner": "run_task09_10_exact.sh", "heavy": True, "outputs": ["data/processed/A.csv"]},
    {"id": "T10", "title": "Preprocessing Version B", "category": "Data", "runner": "run_task09_10_exact.sh", "heavy": True, "outputs": ["data/processed/B.csv"]},
    {"id": "T11", "title": "Train/Validation/Test Split", "category": "Data", "runner": "run_task11_splits.sh", "heavy": False, "outputs": ["reports/task11_split_summary.md"]},
    {"id": "T12", "title": "Experiment Design", "category": "Methodology", "runner": "run_task12_experiment_design.sh", "heavy": False, "outputs": ["configs/task12_experiment_design.json", "reports/task12_experiment_design_summary.md"]},
    {"id": "T13", "title": "Train BERTweet", "category": "Training", "runner": "run_task13_train_bertweet.sh", "heavy": True, "outputs": ["reports/task13/E01_BERTweet_VersionA_metrics.json", "reports/task13/E02_BERTweet_VersionB_metrics.json"]},
    {"id": "T14", "title": "Train RoBERTa", "category": "Training", "runner": "run_task14_train_roberta.sh", "heavy": True, "outputs": ["reports/task14/E03_RoBERTa_VersionA_metrics.json", "reports/task14/E04_RoBERTa_VersionB_metrics.json"]},
    {"id": "T15", "title": "Model Evaluation", "category": "Evaluation", "runner": "run_task15_model_evaluation.sh", "heavy": True, "outputs": ["reports/task15/task15_model_evaluation_summary.md"]},
    {"id": "T16", "title": "Stopword Impact Analysis", "category": "Evaluation", "runner": "run_task16_stopword_impact_analysis.sh", "heavy": False, "outputs": ["reports/task16/task16_stopword_impact_summary.md"]},
    {"id": "T17", "title": "Model Comparison", "category": "Evaluation", "runner": "run_task17_model_comparison.sh", "heavy": False, "outputs": ["reports/task17/task17_model_comparison_summary.md"]},
    {"id": "T18", "title": "Error Analysis", "category": "Interpretation", "runner": "run_task18_error_analysis.sh", "heavy": True, "outputs": ["reports/task18/task18_error_analysis_summary.md"]},
    {"id": "T19", "title": "Final Model Selection", "category": "Finalization", "runner": "run_task19_final_model_selection.sh", "heavy": False, "outputs": ["configs/final_model_config.json", "reports/task19/final_model_card.md"]},
    {"id": "T20", "title": "Original Demo Interface", "category": "Interface", "runner": "run_task20_demo.sh", "heavy": False, "outputs": ["app/streamlit_app.py", "src/predict_sarcasm.py"]},
    {"id": "T20B", "title": "Enhanced Dashboard", "category": "Interface", "runner": "run_enhanced_dashboard.sh", "heavy": False, "outputs": ["app/enhanced_dashboard.py"]},
    {"id": "T21", "title": "Ethics and Limitations", "category": "Responsible AI", "runner": "run_task21_ethics_limitations.sh", "heavy": False, "outputs": ["reports/task21/task21_ethics_and_limitations_summary.md"]},
    {"id": "T22", "title": "Final Documentation", "category": "Documentation", "runner": "run_task22_final_report.sh", "heavy": False, "outputs": ["reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md"]},
    {"id": "T23", "title": "Gradio Interface", "category": "Interface", "runner": "run_gradio_dashboard.sh", "heavy": False, "outputs": ["app/gradio_app.py"]},
]


def rel(path: str | Path) -> Path:
    return PROJECT_ROOT / path


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
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def metric(data: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
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
                pass
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
        rows.append(
            {
                "Experiment": exp,
                "Model": data.get("model_family", "BERTweet" if "BERTweet" in exp else "RoBERTa"),
                "Version": "A" if "VersionA" in exp else "B",
                "Preprocessing": data.get("preprocessing", "Stopwords kept" if "VersionA" in exp else "Selective stopword removal"),
                "Accuracy": round(metric(data, "metrics.accuracy", "accuracy", default=0.0) or 0.0, 4),
                "Macro-F1": round(metric(data, "metrics.macro_f1", "macro_f1", default=0.0) or 0.0, 4),
                "Weighted-F1": round(metric(data, "metrics.weighted_f1", "weighted_f1", default=0.0) or 0.0, 4),
                "Macro Precision": round(metric(data, "metrics.macro_precision", "macro_precision", default=0.0) or 0.0, 4),
                "Macro Recall": round(metric(data, "metrics.macro_recall", "macro_recall", default=0.0) or 0.0, 4),
                "Rows": int(data.get("rows_evaluated", 0) or 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def load_final_model_cards() -> Tuple[str, str, str]:
    final_config = safe_read_json("configs/final_model_config.json")
    summary = safe_read_json("reports/task19/task19_final_model_selection_summary.json")
    metrics_df = load_metrics_table()
    best_acc = "0.7223"
    best_f1 = "0.7167"
    if not metrics_df.empty:
        best_acc = f"{float(metrics_df.iloc[0]['Accuracy']):.4f}"
        best_f1 = f"{float(metrics_df.iloc[0]['Macro-F1']):.4f}"
    selected = summary.get("selected_experiment") or final_config.get("selected_experiment") or "E03_RoBERTa_VersionA"
    return str(selected), best_acc, best_f1


def task_status_df() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        outputs = task["outputs"]
        passed = sum(1 for output in outputs if rel(output).exists())
        completed = passed == len(outputs) and len(outputs) > 0
        rows.append(
            {
                "Task": task["id"],
                "Stage": task["category"],
                "Name": task["title"],
                "Status": "Done" if completed else "Pending / Local Missing",
                "Checks": f"{passed}/{len(outputs)}",
                "Heavy": "Yes" if task.get("heavy") else "No",
            }
        )
    return pd.DataFrame(rows)


def progress_html() -> str:
    df = task_status_df()
    completed = int((df["Status"] == "Done").sum()) if not df.empty else 0
    total = len(df)
    pct = round(completed / total * 100, 1) if total else 0
    selected, acc, f1 = load_final_model_cards()
    return f"""
    <div class='hero'>
        <div>
            <div class='eyebrow'>WQF7007 NLP PROJECT • GROUP 21</div>
            <h1>{APP_TITLE}</h1>
            <p class='hero-text'>An interactive Gradio dashboard for exploring the complete sarcasm detection workflow: preprocessing, model training, evaluation, comparison, error analysis, final selection, ethics, and live prediction.</p>
        </div>
        <div class='hero-grid'>
            <div class='metric-card'><span>Final Model</span><strong>{selected}</strong><small>Stopwords kept • context-aware input</small></div>
            <div class='metric-card'><span>Test Accuracy</span><strong>{acc}</strong><small>Held-out test split</small></div>
            <div class='metric-card'><span>Test Macro-F1</span><strong>{f1}</strong><small>Primary selection metric</small></div>
            <div class='metric-card'><span>Workflow Progress</span><strong>{completed}/{total}</strong><small>{pct}% complete</small></div>
        </div>
    </div>
    <div class='progress-wrap'><div class='progress-bar' style='width:{pct}%;'></div></div>
    """


@lru_cache(maxsize=4)
def get_predictor(device: str = "auto"):
    if SarcasmPredictor is None:
        raise RuntimeError(f"Prediction module could not be imported: {PREDICTOR_IMPORT_ERROR}")
    return SarcasmPredictor(config_path=FINAL_CONFIG_PATH, device=device)


def predict(parent_comment: str, comment: str, device: str = "auto") -> Tuple[str, Dict[str, float], str, Dict[str, Any]]:
    try:
        predictor = get_predictor(device)
        result = predictor.predict(comment=comment, parent_comment=parent_comment)
        result_dict = result.to_dict()
        label = result.label
        confidence = result.confidence
        prob_non = result.probability_non_sarcastic
        prob_sarc = result.probability_sarcastic
        if label.lower().startswith("sarcastic"):
            badge = "<div class='prediction-card sarcastic'><span>Prediction</span><h2>😏 Sarcastic</h2>"
        else:
            badge = "<div class='prediction-card sincere'><span>Prediction</span><h2>🙂 Non-sarcastic</h2>"
        html = f"""
        {badge}
            <p>Confidence: <b>{confidence:.4f}</b></p>
            <div class='prob-row'><span>Non-sarcastic</span><div><i style='width:{prob_non*100:.1f}%'></i></div><b>{prob_non:.4f}</b></div>
            <div class='prob-row'><span>Sarcastic</span><div><i style='width:{prob_sarc*100:.1f}%'></i></div><b>{prob_sarc:.4f}</b></div>
        </div>
        """
        label_scores = {"Non-sarcastic": prob_non, "Sarcastic": prob_sarc}
        return html, label_scores, result.combined_text, result_dict
    except Exception as exc:
        error_html = f"""
        <div class='prediction-card error'>
            <span>Prediction failed</span>
            <h2>⚠️ Model not available</h2>
            <p>{str(exc)}</p>
            <p>Please confirm that <code>models/roberta/versionA</code> exists locally, or update <code>configs/final_model_config.json</code> with a valid Hugging Face model path.</p>
        </div>
        """
        return error_html, {}, "", {"error": str(exc)}


def model_metric_figure(metric_name: str = "Macro-F1"):
    df = load_metrics_table()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if df.empty or metric_name not in df.columns:
        ax.text(0.5, 0.5, "No metric files found", ha="center", va="center")
        ax.axis("off")
        return fig
    labels = [f"{row['Model']} V{row['Version']}" for _, row in df.iterrows()]
    values = df[metric_name].astype(float).tolist()
    ax.bar(labels, values)
    ax.set_title(f"Model comparison by {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, max(1.0, max(values) + 0.08))
    for i, val in enumerate(values):
        ax.text(i, val + 0.015, f"{val:.4f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def stopword_figure():
    df = load_metrics_table()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if df.empty:
        ax.text(0.5, 0.5, "No metric files found", ha="center", va="center")
        ax.axis("off")
        return fig
    pivot = df.pivot_table(index="Model", columns="Version", values="Macro-F1", aggfunc="first")
    pivot = pivot.reindex(["BERTweet", "RoBERTa"])
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Stopword impact on Macro-F1")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Preprocessing Version", labels=["A: stopwords kept", "B: stopwords removed"])
    fig.tight_layout()
    return fig


def confusion_figure(experiment: str):
    path = f"reports/task15/{experiment}_test_metrics.json"
    data = safe_read_json(path)
    cm = data.get("metrics", {}).get("confusion_matrix")
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    if not cm:
        ax.text(0.5, 0.5, "Confusion matrix not found", ha="center", va="center")
        ax.axis("off")
        return fig
    matrix = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    im = ax.imshow(matrix.values)
    ax.set_xticks(range(2), matrix.columns)
    ax.set_yticks(range(2), matrix.index)
    ax.set_title(experiment)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix.iloc[i, j]:,}", ha="center", va="center", color="white" if matrix.iloc[i, j] > matrix.values.max()/2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def report_files() -> List[str]:
    rows: List[str] = []
    for base in [PROJECT_ROOT / "reports", PROJECT_ROOT / "docs"]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if "local_only" in path.parts or "error_samples" in path.parts or "ui_runs" in path.parts:
                continue
            if path.suffix.lower() in {".md", ".txt", ".json", ".csv"}:
                rows.append(str(path.relative_to(PROJECT_ROOT)))
    return rows


def load_report(path: str) -> Tuple[str, pd.DataFrame]:
    if not path:
        return "Select a report file.", pd.DataFrame()
    p = rel(path)
    if not p.exists():
        return f"File not found: {path}", pd.DataFrame()
    suffix = p.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(p)
            return f"### {path}\n\nCSV loaded below.", df
        except Exception as exc:
            return f"Could not read CSV: {exc}", pd.DataFrame()
    text = p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        try:
            parsed = json.loads(text)
            text = "```json\n" + json.dumps(parsed, indent=2)[:15000] + "\n```"
        except Exception:
            text = "```\n" + text[:15000] + "\n```"
    else:
        text = text[:15000]
    if len(text) >= 15000:
        text += "\n\n---\nPreview truncated in the UI. Open the file in the repository for full content."
    return f"### {path}\n\n{text}", pd.DataFrame()


def runner_choices() -> List[str]:
    return [f"{task['id']} - {task['title']}" for task in TASKS if task.get("runner")]


def task_by_label(label: str) -> Optional[Dict[str, Any]]:
    task_id = label.split(" - ", 1)[0].strip() if label else ""
    for task in TASKS:
        if task["id"] == task_id:
            return task
    return None


def safe_log_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def run_task(label: str, confirm_heavy: bool, quick_mode: bool) -> Iterable[str]:
    allow_runs = os.getenv("GRADIO_ALLOW_TASK_RUNS", "1").lower() in {"1", "true", "yes", "y"}
    if not allow_runs:
        yield "Task execution is disabled. Set GRADIO_ALLOW_TASK_RUNS=1 for local use."
        return
    task = task_by_label(label)
    if not task:
        yield "Please select a valid task."
        return
    if task.get("heavy") and not confirm_heavy:
        yield "This is a heavy/local task. Tick the confirmation box before running it."
        return
    runner = task.get("runner")
    if not runner:
        yield "No runner script configured for this task."
        return
    runner_path = rel(runner)
    if not runner_path.exists():
        yield f"Runner script not found: {runner}"
        return
    cmd = ["bash", str(runner_path)] if str(runner).endswith(".sh") else [str(runner_path)]
    env = os.environ.copy()
    if quick_mode:
        env.update(
            {
                "TASK13_MAX_TRAIN_SAMPLES": "1000",
                "TASK13_MAX_VALID_SAMPLES": "300",
                "TASK14_MAX_TRAIN_SAMPLES": "1000",
                "TASK14_MAX_VALID_SAMPLES": "300",
                "TASK15_MAX_TEST_SAMPLES": "1000",
            }
        )
    log_path = UI_RUNS_DIR / f"{safe_log_name(task['id'])}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    yield f"Starting {task['id']} - {task['title']}\nCommand: {' '.join(cmd)}\nLog: {log_path}\n"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        buffer = ""
        for line in proc.stdout:
            log.write(line)
            log.flush()
            buffer += line
            if len(buffer) > 20000:
                buffer = buffer[-20000:]
            yield buffer
        return_code = proc.wait()
        log.write(f"\nReturn code: {return_code}\n")
        buffer += f"\nFinished with return code {return_code}.\n"
        yield buffer


def ethics_text() -> str:
    content = safe_read_text("reports/task21/task21_ethics_and_limitations_summary.md")
    if not content:
        content = safe_read_text("reports/task19/final_model_card.md")
    if not content:
        return "Ethics report not found. Run Task 21 first."
    return content[:12000]


def hosting_text() -> str:
    return """
    ## Hosting readiness

    The current local Gradio app can run directly from the project repository. For Hugging Face deployment, the recommended setup is:

    1. Upload the final `models/roberta/versionA` checkpoint to a Hugging Face Model repository.
    2. Set `configs/final_model_config.json` or environment variables so the app loads from the Hugging Face model path.
    3. Use the root `app.py` file as the Hugging Face Spaces entry point.
    4. Disable task execution on the public hosted app by setting `GRADIO_ALLOW_TASK_RUNS=0`.

    Heavy training scripts should remain local because public demo spaces are mainly for inference, result browsing, and report presentation.
    """


def custom_css() -> str:
    return f"""
    .gradio-container {{
        background: radial-gradient(circle at top left, #eef2ff 0%, #f8fafc 35%, #ffffff 100%) !important;
        color: {DARK_COLOR};
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(6,182,212,0.92));
        border-radius: 28px;
        padding: 34px;
        color: white;
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 24px;
        box-shadow: 0 24px 60px rgba(15,23,42,0.18);
        margin-bottom: 18px;
    }}
    .hero h1 {{
        font-size: 44px;
        line-height: 1.04;
        margin: 8px 0 12px;
        letter-spacing: -0.04em;
    }}
    .hero-text {{
        font-size: 16px;
        max-width: 780px;
        opacity: 0.95;
    }}
    .eyebrow {{
        letter-spacing: 0.12em;
        font-size: 12px;
        text-transform: uppercase;
        font-weight: 800;
        opacity: 0.86;
    }}
    .hero-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
    }}
    .metric-card {{
        background: rgba(255,255,255,0.17);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 20px;
        padding: 18px;
        backdrop-filter: blur(8px);
    }}
    .metric-card span {{ display: block; font-size: 12px; opacity: .85; text-transform: uppercase; letter-spacing: .08em; }}
    .metric-card strong {{ display: block; font-size: 24px; margin: 7px 0; }}
    .metric-card small {{ opacity: .85; }}
    .progress-wrap {{ height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-bottom: 20px; }}
    .progress-bar {{ height: 100%; background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); border-radius: 999px; }}
    .prediction-card {{ border-radius: 24px; padding: 24px; border: 1px solid rgba(148,163,184,0.3); box-shadow: 0 18px 45px rgba(15,23,42,.08); }}
    .prediction-card span {{ text-transform: uppercase; font-size: 12px; letter-spacing: .10em; font-weight: 800; color: #64748b; }}
    .prediction-card h2 {{ font-size: 34px; margin: 8px 0; }}
    .prediction-card.sarcastic {{ background: linear-gradient(135deg, #fff7ed, #ffedd5); }}
    .prediction-card.sincere {{ background: linear-gradient(135deg, #ecfeff, #e0f2fe); }}
    .prediction-card.error {{ background: linear-gradient(135deg, #fef2f2, #fee2e2); }}
    .prob-row {{ display: grid; grid-template-columns: 130px 1fr 72px; gap: 12px; align-items: center; margin: 12px 0; }}
    .prob-row div {{ height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .prob-row i {{ display: block; height: 100%; background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); border-radius: 999px; }}
    .small-note {{ color: #64748b; font-size: 13px; }}
    @media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; }} .hero-grid {{ grid-template-columns: 1fr; }} }}
    """


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(primary_hue="violet", secondary_hue="cyan"), css=custom_css()) as demo:
        gr.HTML(progress_html())

        with gr.Tabs():
            with gr.Tab("🔮 Live Sarcasm Detector"):
                gr.Markdown("### Try the final model with optional Reddit conversation context")
                with gr.Row():
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
                        pred_html = gr.HTML("<div class='prediction-card'><span>Waiting</span><h2>Enter text and predict</h2><p>The final RoBERTa Version A model will return class probabilities and confidence.</p></div>")
                        label_output = gr.Label(label="Class probabilities")
                        combined_text = gr.Textbox(label="Combined model input", lines=4)
                        json_output = gr.JSON(label="Technical output")
                predict_btn.click(predict, inputs=[parent, comment, device], outputs=[pred_html, label_output, combined_text, json_output])

            with gr.Tab("🧭 Project Workflow"):
                gr.Markdown("### End-to-end NLP workflow status")
                refresh_status = gr.Button("Refresh workflow status")
                status_df = gr.Dataframe(value=task_status_df(), label="Task completion matrix", interactive=False, wrap=True)
                refresh_status.click(task_status_df, outputs=status_df)
                gr.Markdown(
                    """
                    The workflow begins with preprocessing and controlled splitting, then moves into experiment design, model training, full evaluation, stopword impact analysis, error analysis, final model selection, UI, documentation, and responsible-use reporting.
                    """
                )

            with gr.Tab("📊 Results Dashboard"):
                gr.Markdown("### Model comparison and stopword impact")
                with gr.Row():
                    metrics_df = gr.Dataframe(value=load_metrics_table(), label="Held-out test metrics", interactive=False, wrap=True)
                with gr.Row():
                    metric_selector = gr.Dropdown(["Macro-F1", "Accuracy", "Weighted-F1", "Macro Precision", "Macro Recall"], value="Macro-F1", label="Metric chart")
                    metric_plot = gr.Plot(value=model_metric_figure("Macro-F1"), label="Metric comparison")
                metric_selector.change(model_metric_figure, inputs=metric_selector, outputs=metric_plot)
                with gr.Row():
                    stopword_plot = gr.Plot(value=stopword_figure(), label="Stopword impact")
                with gr.Row():
                    cm_choice = gr.Dropdown(
                        ["E03_RoBERTa_VersionA", "E04_RoBERTa_VersionB", "E01_BERTweet_VersionA", "E02_BERTweet_VersionB"],
                        value="E03_RoBERTa_VersionA",
                        label="Confusion matrix experiment",
                    )
                    cm_plot = gr.Plot(value=confusion_figure("E03_RoBERTa_VersionA"), label="Confusion matrix")
                cm_choice.change(confusion_figure, inputs=cm_choice, outputs=cm_plot)
                gr.Markdown("**Current final choice:** RoBERTa Version A, because it achieved the strongest held-out test Macro-F1 while preserving stopwords and conversational context.")

            with gr.Tab("📚 Reports Explorer"):
                gr.Markdown("### Browse generated reports and documentation")
                report_dropdown = gr.Dropdown(choices=report_files(), label="Report file", value="reports/task15/task15_model_evaluation_summary.md" if rel("reports/task15/task15_model_evaluation_summary.md").exists() else None)
                with gr.Row():
                    load_report_btn = gr.Button("Open report", variant="primary")
                    refresh_reports_btn = gr.Button("Refresh file list")
                report_md = gr.Markdown("Select a report and click **Open report**.")
                report_table = gr.Dataframe(label="CSV preview", interactive=False, wrap=True)
                load_report_btn.click(load_report, inputs=report_dropdown, outputs=[report_md, report_table])
                refresh_reports_btn.click(lambda: gr.Dropdown(choices=report_files()), outputs=report_dropdown)

            with gr.Tab("⚙️ Local Pipeline Runner"):
                gr.Markdown(
                    """
                    ### Run project scripts from the UI

                    This section is intended for **local use**. Training and evaluation can take time. For Hugging Face hosting, set `GRADIO_ALLOW_TASK_RUNS=0` and keep this tab as a read-only explanation.
                    """
                )
                task_select = gr.Dropdown(choices=runner_choices(), label="Task runner", value="T17 - Model Comparison" if "T17 - Model Comparison" in runner_choices() else None)
                confirm_heavy = gr.Checkbox(label="I understand this may run a heavy/local task", value=False)
                quick_mode = gr.Checkbox(label="Quick mode for supported heavy tasks", value=True)
                run_btn = gr.Button("Run selected task", variant="primary")
                run_log = gr.Textbox(label="Live task log", lines=22, max_lines=35)
                run_btn.click(run_task, inputs=[task_select, confirm_heavy, quick_mode], outputs=run_log)

            with gr.Tab("🧪 Error Analysis & Ethics"):
                gr.Markdown("### Error analysis and responsible use")
                with gr.Accordion("Task 18 error analysis", open=True):
                    gr.Markdown(safe_read_text("reports/task18/task18_error_analysis_summary.md", "Task 18 report not found."))
                with gr.Accordion("Task 21 ethics and limitations", open=False):
                    gr.Markdown(ethics_text())

            with gr.Tab("🚀 Hosting Readiness"):
                gr.Markdown(hosting_text())
                gr.Markdown(
                    """
                    ### Important files for Gradio hosting

                    - `app/gradio_app.py` — main Gradio Blocks implementation.
                    - `app.py` — Hugging Face Spaces entry point.
                    - `run_gradio_dashboard.sh` — local macOS/Linux runner.
                    - `requirements.txt` — includes Gradio and ML dependencies.
                    - `configs/final_model_config.json` — selected final model configuration.
                    """
                )
    return demo


demo = build_app()


def main() -> None:
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    main()
