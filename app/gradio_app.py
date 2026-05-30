#!/usr/bin/env python3
"""Polished Gradio interface for the sarcasm detection NLP project.

Task 23C fixes the previous Gradio UI problems:
- readable typography and colors
- no raw JSON/config dumps in visible cards
- stable Results tab without matplotlib crashes
- clean light dashboard layout
- stakeholder-facing interface with Gradio as the primary UI
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
from typing import Any, Dict, List, Tuple

import gradio as gr
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

APP_TITLE = "Sarcasm Detection NLP Dashboard"
FINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.json"
UI_RUNS_DIR = PROJECT_ROOT / "reports" / "ui_runs"
UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)

TASKS: List[Dict[str, Any]] = [
    {"id": "T09", "stage": "Data", "name": "Preprocessing Version A", "runner": "run_task09_10_exact.sh", "heavy": True, "outputs": ["data/processed/A.csv", "docs/task09_10_exact_reproduction_note.md"]},
    {"id": "T10", "stage": "Data", "name": "Preprocessing Version B", "runner": "run_task09_10_exact.sh", "heavy": True, "outputs": ["data/processed/B.csv", "docs/task09_10_exact_reproduction_note.md"]},
    {"id": "T11", "stage": "Data", "name": "Train/Validation/Test Split", "runner": "run_task11_splits.sh", "heavy": False, "outputs": ["reports/task11_split_summary.md"]},
    {"id": "T12", "stage": "Methodology", "name": "Experiment Design", "runner": "run_task12_experiment_design.sh", "heavy": False, "outputs": ["configs/task12_experiment_design.json"]},
    {"id": "T13", "stage": "Training", "name": "Train BERTweet", "runner": "run_task13_train_bertweet.sh", "heavy": True, "outputs": ["reports/task13/E01_BERTweet_VersionA_metrics.json", "reports/task13/E02_BERTweet_VersionB_metrics.json"]},
    {"id": "T14", "stage": "Training", "name": "Train RoBERTa", "runner": "run_task14_train_roberta.sh", "heavy": True, "outputs": ["reports/task14/E03_RoBERTa_VersionA_metrics.json", "reports/task14/E04_RoBERTa_VersionB_metrics.json"]},
    {"id": "T15", "stage": "Evaluation", "name": "Full Test Evaluation", "runner": "run_task15_model_evaluation.sh", "heavy": True, "outputs": ["reports/task15/task15_model_evaluation_summary.md"]},
    {"id": "T16", "stage": "Evaluation", "name": "Stopword Impact Analysis", "runner": "run_task16_stopword_impact_analysis.sh", "heavy": False, "outputs": ["reports/task16/task16_stopword_impact_summary.md"]},
    {"id": "T17", "stage": "Evaluation", "name": "Model Comparison", "runner": "run_task17_model_comparison.sh", "heavy": False, "outputs": ["reports/task17/task17_model_comparison_summary.md"]},
    {"id": "T18", "stage": "Interpretation", "name": "Error Analysis", "runner": "run_task18_error_analysis.sh", "heavy": True, "outputs": ["reports/task18/task18_error_analysis_summary.md"]},
    {"id": "T19", "stage": "Finalization", "name": "Final Model Selection", "runner": "run_task19_final_model_selection.sh", "heavy": False, "outputs": ["configs/final_model_config.json", "reports/task19/final_model_card.md"]},
    {"id": "T20", "stage": "Interface", "name": "Initial Demo", "runner": "run_task20_demo.sh", "heavy": False, "outputs": ["src/predict_sarcasm.py"]},
    {"id": "T21", "stage": "Responsible AI", "name": "Ethics and Limitations", "runner": "run_task21_ethics_limitations.sh", "heavy": False, "outputs": ["reports/task21/task21_ethics_and_limitations_summary.md"]},
    {"id": "T22", "stage": "Documentation", "name": "Final Documentation", "runner": "run_task22_final_report.sh", "heavy": False, "outputs": ["reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md"]},
    {"id": "T23", "stage": "Interface", "name": "Gradio Interface", "runner": "run_gradio_dashboard.sh", "heavy": False, "outputs": ["app/gradio_app.py"]},
]

DEFAULT_RESULTS = [
    {"Experiment": "E03_RoBERTa_VersionA", "Model": "RoBERTa", "Version": "A", "Preprocessing": "Stopwords kept", "Accuracy": 0.7223, "Macro-F1": 0.7167, "Weighted-F1": 0.7165, "Rows": 96509},
    {"Experiment": "E04_RoBERTa_VersionB", "Model": "RoBERTa", "Version": "B", "Preprocessing": "Stopwords removed", "Accuracy": 0.6773, "Macro-F1": 0.6648, "Weighted-F1": 0.6644, "Rows": 96509},
    {"Experiment": "E01_BERTweet_VersionA", "Model": "BERTweet", "Version": "A", "Preprocessing": "Stopwords kept", "Accuracy": 0.5092, "Macro-F1": 0.3632, "Weighted-F1": 0.3615, "Rows": 96509},
    {"Experiment": "E02_BERTweet_VersionB", "Model": "BERTweet", "Version": "B", "Preprocessing": "Stopwords removed", "Accuracy": 0.5018, "Macro-F1": 0.3452, "Weighted-F1": 0.3435, "Rows": 96509},
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
        value = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def nested_get(data: Dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur: Any = data
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, "", {}):
            return cur
    return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metric(data: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = nested_get(data, key, default=None)
        if value is not None:
            return to_float(value, default)
    return default


def clean_model_name(value: Any) -> str:
    text = str(value or "RoBERTa V-A")
    if "RoBERTa" in text or "roberta" in text.lower():
        return "RoBERTa V-A" if "VersionB" not in text and "Version B" not in text else "RoBERTa V-B"
    if "BERTweet" in text or "bertweet" in text.lower():
        return "BERTweet V-A" if "VersionB" not in text and "Version B" not in text else "BERTweet V-B"
    return "RoBERTa V-A"


def load_metrics_table() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_files = {
        "E01_BERTweet_VersionA": "reports/task15/E01_BERTweet_VersionA_test_metrics.json",
        "E02_BERTweet_VersionB": "reports/task15/E02_BERTweet_VersionB_test_metrics.json",
        "E03_RoBERTa_VersionA": "reports/task15/E03_RoBERTa_VersionA_test_metrics.json",
        "E04_RoBERTa_VersionB": "reports/task15/E04_RoBERTa_VersionB_test_metrics.json",
    }
    for exp, path in metric_files.items():
        data = safe_read_json(path)
        if not data:
            continue
        model = "BERTweet" if "BERTweet" in exp else "RoBERTa"
        version = "A" if "VersionA" in exp else "B"
        rows.append({
            "Experiment": exp,
            "Model": model,
            "Version": version,
            "Preprocessing": "Stopwords kept" if version == "A" else "Stopwords removed",
            "Accuracy": round(metric(data, "metrics.accuracy", "accuracy"), 4),
            "Macro-F1": round(metric(data, "metrics.macro_f1", "macro_f1"), 4),
            "Weighted-F1": round(metric(data, "metrics.weighted_f1", "weighted_f1"), 4),
            "Macro Precision": round(metric(data, "metrics.macro_precision", "macro_precision"), 4),
            "Macro Recall": round(metric(data, "metrics.macro_recall", "macro_recall"), 4),
            "Rows": int(data.get("rows_evaluated", data.get("test_rows", 0)) or 0),
        })
    if not rows:
        rows = [dict(row) for row in DEFAULT_RESULTS]
    df = pd.DataFrame(rows)
    df = df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    if "Rank" not in df.columns:
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def final_info() -> Dict[str, Any]:
    df = load_metrics_table()
    top = df.iloc[0].to_dict() if not df.empty else DEFAULT_RESULTS[0]
    return {
        "model": clean_model_name(top.get("Experiment")),
        "preprocessing": str(top.get("Preprocessing", "Stopwords kept")),
        "accuracy": to_float(top.get("Accuracy", 0.7223), 0.7223),
        "macro_f1": to_float(top.get("Macro-F1", 0.7167), 0.7167),
        "weighted_f1": to_float(top.get("Weighted-F1", 0.7165), 0.7165),
        "rows": int(top.get("Rows", 96509) or 96509),
        "experiment": str(top.get("Experiment", "E03_RoBERTa_VersionA")),
    }


def task_status_rows() -> List[Dict[str, Any]]:
    rows = []
    for task in TASKS:
        outputs = task["outputs"]
        passed = sum(1 for item in outputs if rel(item).exists())
        done = passed == len(outputs) and len(outputs) > 0
        rows.append({
            "Task": task["id"],
            "Stage": task["stage"],
            "Name": task["name"],
            "Status": "Done" if done else "Pending local file",
            "Checks": f"{passed}/{len(outputs)}",
            "Heavy": "Yes" if task.get("heavy") else "No",
        })
    return rows


def workflow_progress() -> Tuple[int, int, float]:
    rows = task_status_rows()
    total = len(rows)
    completed = sum(1 for row in rows if row["Status"] == "Done")
    pct = (completed / total * 100.0) if total else 0.0
    return completed, total, pct


def html_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<div class='empty'>No rows available.</div>"
    headers = list(rows[0].keys())
    thead = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for h in headers:
            val = row.get(h, "")
            if h == "Status":
                cls = "status-done" if str(val).lower().startswith("done") else "status-warn"
                cells.append(f"<td><span class='{cls}'>{esc(val)}</span></td>")
            else:
                cells.append(f"<td>{esc(val)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table class='clean-table'><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def metric_cards() -> str:
    info = final_info()
    completed, total, pct = workflow_progress()
    return f"""
    <div class="metric-grid">
      <div class="metric-card"><span>Final model</span><strong>{esc(info['model'])}</strong><small>{esc(info['preprocessing'])}</small></div>
      <div class="metric-card"><span>Test accuracy</span><strong>{info['accuracy']:.4f}</strong><small>Held-out test split</small></div>
      <div class="metric-card"><span>Test Macro-F1</span><strong>{info['macro_f1']:.4f}</strong><small>Primary selection metric</small></div>
      <div class="metric-card"><span>Workflow</span><strong>{completed}/{total}</strong><small>{pct:.1f}% completed</small></div>
    </div>
    """


def hero_html() -> str:
    completed, total, pct = workflow_progress()
    return f"""
    <section class="hero">
      <div class="hero-left">
        <div class="kicker">WQF7007 NLP Project • Group 21</div>
        <h1>Sarcasm Detection<br><span>NLP Dashboard</span></h1>
        <p>Explore the complete Gradio-based sarcasm detection workflow: preprocessing, transformer training, evaluation, model comparison, reports, ethics, and live prediction.</p>
        <div class="chips">
          <span>Gradio UI</span><span>RoBERTa final model</span><span>Context-aware input</span><span>Stopwords kept</span>
        </div>
      </div>
      <div class="hero-right">
        {metric_cards()}
      </div>
    </section>
    <div class="progress"><i style="width:{pct:.1f}%"></i></div>
    """


def overview_html() -> str:
    info = final_info()
    return f"""
    <div class="content-grid">
      <div class="panel">
        <h2>Project snapshot</h2>
        <p>This dashboard presents the complete sarcasm detection project in one place. The study compares BERTweet and RoBERTa under two preprocessing settings: Version A keeps stopwords, while Version B selectively removes stopwords but preserves important negations.</p>
        <p>The selected final model is <b>{esc(info['model'])}</b> using <b>{esc(info['preprocessing'])}</b>. It achieved <b>{info['accuracy']:.4f}</b> accuracy and <b>{info['macro_f1']:.4f}</b> Macro-F1 on the held-out test split.</p>
        <h3>What users can do here</h3>
        <ol>
          <li>Run live sarcasm predictions with optional Reddit parent-comment context.</li>
          <li>View all four model results and stopword impact comparisons.</li>
          <li>Browse generated reports without leaving the dashboard.</li>
          <li>Check workflow completion and local task runners.</li>
          <li>Review ethics, limitations, and hosting readiness.</li>
        </ol>
      </div>
      <div class="panel highlight">
        <h2>Final model</h2>
        <div class="big-number">{esc(info['model'])}</div>
        <p>{esc(info['preprocessing'])}</p>
        <div class="mini-list">
          <div><span>Accuracy</span><b>{info['accuracy']:.4f}</b></div>
          <div><span>Macro-F1</span><b>{info['macro_f1']:.4f}</b></div>
          <div><span>Test rows</span><b>{info['rows']:,}</b></div>
        </div>
      </div>
    </div>
    <div class="panel"><h2>Workflow status</h2>{html_table(task_status_rows())}</div>
    """


@lru_cache(maxsize=4)
def get_predictor(device: str = "auto"):
    if SarcasmPredictor is None:
        raise RuntimeError(f"Prediction module could not be imported: {PREDICTOR_IMPORT_ERROR}")
    return SarcasmPredictor(config_path=FINAL_CONFIG_PATH, device=device)


def waiting_card() -> str:
    return """
    <div class="prediction-card waiting">
      <span class="badge">Waiting for input</span>
      <h2>Enter text to classify</h2>
      <p>Add a Reddit reply, optionally with parent-comment context, then run the final model.</p>
      <div class="quick-meta"><span>Model: RoBERTa V-A</span><span>Max length: 128</span></div>
    </div>
    """


def probability_bars(prob_non: float = 0.0, prob_sarc: float = 0.0) -> str:
    return f"""
    <div class="prob-box">
      <h3>Class probabilities</h3>
      <div class="prob-row"><div><span>Non-sarcastic</span><b>{prob_non:.4f}</b></div><div class="bar"><i style="width:{prob_non*100:.1f}%"></i></div></div>
      <div class="prob-row"><div><span>Sarcastic</span><b>{prob_sarc:.4f}</b></div><div class="bar alt"><i style="width:{prob_sarc*100:.1f}%"></i></div></div>
    </div>
    """


def predict(parent_comment: str, comment: str, device: str) -> Tuple[str, str, str, str]:
    try:
        predictor = get_predictor(device)
        result = predictor.predict(comment=comment, parent_comment=parent_comment)
        prob_non = float(result.probability_non_sarcastic)
        prob_sarc = float(result.probability_sarcastic)
        confidence = float(result.confidence)
        is_sarc = result.label.lower().startswith("sarcastic")
        verdict = "Sarcastic" if is_sarc else "Non-sarcastic"
        emoji = "😏" if is_sarc else "🙂"
        tone = "sarcastic" if is_sarc else "neutral"
        note = "The model detected sarcastic intent in the reply." if is_sarc else "The model interpreted the reply as sincere or literal."
        card = f"""
        <div class="prediction-card {tone}">
          <span class="badge">Prediction result</span>
          <h2>{emoji} {verdict}</h2>
          <p>{note}</p>
          <div class="confidence"><span>Confidence</span><b>{confidence:.4f}</b></div>
        </div>
        """
        details = f"Label: {verdict}\nConfidence: {confidence:.4f}\nP(non-sarcastic): {prob_non:.4f}\nP(sarcastic): {prob_sarc:.4f}\nModel: {result.model_name}\nCheckpoint: {result.checkpoint_path}"
        return card, probability_bars(prob_non, prob_sarc), result.combined_text, details
    except Exception as exc:
        return f"""
        <div class="prediction-card error">
          <span class="badge">Prediction unavailable</span>
          <h2>Model could not run</h2>
          <p>{esc(exc)}</p>
        </div>
        """, probability_bars(), "", str(exc)


def results_html() -> str:
    df = load_metrics_table()
    rows = df.to_dict("records")
    info = final_info()
    roberta_a = float(df[df["Experiment"].eq("E03_RoBERTa_VersionA")]["Macro-F1"].iloc[0]) if "E03_RoBERTa_VersionA" in set(df["Experiment"]) else 0.7167
    roberta_b = float(df[df["Experiment"].eq("E04_RoBERTa_VersionB")]["Macro-F1"].iloc[0]) if "E04_RoBERTa_VersionB" in set(df["Experiment"]) else 0.6648
    bertweet_a = float(df[df["Experiment"].eq("E01_BERTweet_VersionA")]["Macro-F1"].iloc[0]) if "E01_BERTweet_VersionA" in set(df["Experiment"]) else 0.3632
    bertweet_b = float(df[df["Experiment"].eq("E02_BERTweet_VersionB")]["Macro-F1"].iloc[0]) if "E02_BERTweet_VersionB" in set(df["Experiment"]) else 0.3452

    def bar(label: str, value: float, color: str) -> str:
        width = max(2, min(100, value * 100))
        return f"<div class='metric-bar'><div><span>{esc(label)}</span><b>{value:.4f}</b></div><em><i style='width:{width:.1f}%;background:{color}'></i></em></div>"

    return f"""
    <div class="panel">
      <h2>Model comparison</h2>
      <p>All four experiments were evaluated on the same held-out test split. Macro-F1 is the main selection metric because it balances both sarcastic and non-sarcastic classes.</p>
      {html_table(rows)}
    </div>
    <div class="content-grid">
      <div class="panel">
        <h2>Macro-F1 ranking</h2>
        {bar('RoBERTa Version A', roberta_a, '#2563eb')}
        {bar('RoBERTa Version B', roberta_b, '#06b6d4')}
        {bar('BERTweet Version A', bertweet_a, '#7c3aed')}
        {bar('BERTweet Version B', bertweet_b, '#a855f7')}
      </div>
      <div class="panel highlight">
        <h2>Final decision</h2>
        <div class="big-number">{esc(info['model'])}</div>
        <p>The best model is <b>{esc(info['model'])}</b> because it achieved the highest held-out test Macro-F1 of <b>{info['macro_f1']:.4f}</b> and accuracy of <b>{info['accuracy']:.4f}</b>.</p>
        <p>The results also showed that keeping stopwords performed better than selective stopword removal for both model families.</p>
      </div>
    </div>
    <div class="panel">
      <h2>Stopword impact</h2>
      {bar('RoBERTa stopwords kept', roberta_a, '#16a34a')}
      {bar('RoBERTa stopwords removed', roberta_b, '#ef4444')}
      {bar('BERTweet stopwords kept', bertweet_a, '#16a34a')}
      {bar('BERTweet stopwords removed', bertweet_b, '#ef4444')}
    </div>
    """


def workflow_html() -> str:
    return f"<div class='panel'><h2>End-to-end workflow</h2><p>This table checks the major project tasks from preprocessing to final documentation and Gradio interface.</p>{html_table(task_status_rows())}</div>"


def report_files() -> List[str]:
    roots = [PROJECT_ROOT / "reports", PROJECT_ROOT / "docs"]
    files: List[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".csv"}:
                files.append(str(path.relative_to(PROJECT_ROOT)))
    preferred = "reports/task15/task15_model_evaluation_summary.md"
    files = sorted(files)
    if preferred in files:
        files.remove(preferred)
        files.insert(0, preferred)
    return files


def load_report(path: str) -> str:
    if not path:
        return "Select a report file."
    p = rel(path)
    if not p.exists() or p.is_dir():
        return f"Report not found: `{path}`"
    text = p.read_text(encoding="utf-8", errors="replace")
    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2)[:12000]
        except Exception:
            text = text[:12000]
        return f"### {esc(path)}\n```json\n{text}\n```"
    if suffix == ".csv":
        try:
            df = pd.read_csv(p)
            return f"### {esc(path)}\n" + df.head(20).to_markdown(index=False)
        except Exception:
            return f"### {esc(path)}\n```text\n{text[:12000]}\n```"
    return text[:12000]


def runner_choices() -> List[str]:
    return [f"{task['id']} — {task['name']} ({task['runner']})" for task in TASKS if rel(task["runner"]).exists()]


def run_task(choice: str, confirm: bool, quick: bool) -> str:
    if os.getenv("GRADIO_ALLOW_TASK_RUNS", "1") not in {"1", "true", "yes"}:
        return "Task execution is disabled for this dashboard."
    if not choice:
        return "Select a task first."
    task_id = choice.split(" — ", 1)[0]
    task = next((item for item in TASKS if item["id"] == task_id), None)
    if not task:
        return "Task not found."
    if task.get("heavy") and not confirm:
        return "This is a heavy task. Tick the confirmation box before running it."
    runner = rel(task["runner"])
    if not runner.exists():
        return f"Runner not found: {task['runner']}"
    env = os.environ.copy()
    if quick:
        env.update({
            "TASK13_MAX_TRAIN_SAMPLES": "128",
            "TASK13_MAX_VALID_SAMPLES": "128",
            "TASK14_MAX_TRAIN_SAMPLES": "128",
            "TASK14_MAX_VALID_SAMPLES": "128",
            "TASK15_MAX_TEST_SAMPLES": "2000",
        })
    start = time.strftime("%Y%m%d_%H%M%S")
    log_file = UI_RUNS_DIR / f"{task_id}_{start}.log"
    process = subprocess.run(["bash", str(runner)], cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=7200)
    log_file.write_text(process.stdout, encoding="utf-8")
    status = "completed" if process.returncode == 0 else f"failed with code {process.returncode}"
    return f"Task {task_id} {status}.\nLog saved to {log_file.relative_to(PROJECT_ROOT)}\n\n" + process.stdout[-6000:]


def ethics_html() -> str:
    summary = safe_read_text("reports/task21/task21_ethics_and_limitations_summary.md", "")
    return f"""
    <div class="content-grid">
      <div class="panel">
        <h2>Responsible-use position</h2>
        <p>This system is a coursework and research demonstration. It should not be used to automatically remove, penalize, or judge user content.</p>
        <p>Sarcasm can depend on culture, speaker intent, conversation history, and external context. The model produces probability-based predictions, not guaranteed truth.</p>
      </div>
      <div class="panel highlight">
        <h2>Hosting readiness</h2>
        <p><b>Recommended:</b> Hugging Face Spaces with Gradio.</p>
        <p>For public hosting, keep task execution disabled and load the final model from a Hugging Face model repository instead of a local checkpoint.</p>
      </div>
    </div>
    <div class="panel"><h2>Task 21 summary</h2><pre class="report-pre">{esc(summary[:5000])}</pre></div>
    """


def custom_css() -> str:
    return """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --subtle: #e2e8f0;
      --blue: #2563eb;
      --cyan: #06b6d4;
      --violet: #7c3aed;
      --green: #16a34a;
      --red: #dc2626;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    }

    html, body, .gradio-container {
      background: var(--bg) !important;
      color: var(--ink) !important;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
      font-size: 16px !important;
    }

    .gradio-container {
      max-width: 1320px !important;
      margin: 0 auto !important;
      padding: 28px 24px 56px !important;
    }

    * {
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
      box-sizing: border-box;
    }

    h1, h2, h3, h4, p, li, span, label, button, input, textarea, select, table, th, td {
      color: var(--ink) !important;
      letter-spacing: normal !important;
      opacity: 1 !important;
    }

    p, li { line-height: 1.65 !important; color: var(--muted) !important; }
    label, .label-wrap span { color: var(--ink) !important; font-weight: 700 !important; }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(380px, 0.9fr);
      gap: 30px;
      align-items: center;
      padding: 44px;
      border-radius: 30px;
      background:
        radial-gradient(circle at 15% 15%, rgba(255,255,255,0.14), transparent 32%),
        linear-gradient(135deg, #312e81 0%, #2563eb 56%, #0891b2 100%);
      box-shadow: var(--shadow);
      margin-bottom: 22px;
      overflow: hidden;
    }

    .hero .kicker {
      color: #a5f3fc !important;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.22em !important;
      font-weight: 900;
      margin-bottom: 18px;
    }

    .hero h1 {
      color: white !important;
      font-size: clamp(42px, 5.2vw, 76px);
      line-height: 0.95;
      letter-spacing: -0.055em !important;
      margin: 0 0 22px;
      font-weight: 900;
    }

    .hero h1 span { color: #cffafe !important; }
    .hero p { color: rgba(255,255,255,0.90) !important; max-width: 720px; font-size: 18px; margin-bottom: 22px; }

    .chips { display: flex; flex-wrap: wrap; gap: 10px; }
    .chips span {
      color: white !important;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(255,255,255,0.13);
      border-radius: 999px;
      padding: 9px 14px;
      font-weight: 700;
      backdrop-filter: blur(8px);
    }

    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .metric-card {
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(255,255,255,0.24);
      border-radius: 22px;
      padding: 20px;
      min-height: 126px;
      backdrop-filter: blur(10px);
    }
    .metric-card span { display:block; color:#dbeafe !important; text-transform:uppercase; letter-spacing:0.14em !important; font-weight:900; font-size:12px; margin-bottom:10px; }
    .metric-card strong { display:block; color:white !important; font-size:32px; font-weight:900; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .metric-card small { color:#e0f2fe !important; font-weight:600; }

    .progress { height: 10px; border-radius: 999px; background:#e2e8f0; overflow:hidden; margin-bottom:22px; }
    .progress i { display:block; height:100%; background:linear-gradient(90deg, var(--violet), var(--blue), var(--cyan)); border-radius:inherit; }

    .tabs { border: 0 !important; }
    .tab-nav { border-bottom: 1px solid #cbd5e1 !important; margin-bottom: 22px !important; }
    .tab-nav button, .tabs button {
      color: #334155 !important;
      background: transparent !important;
      font-weight: 800 !important;
      font-size: 15px !important;
      border-radius: 14px 14px 0 0 !important;
      opacity: 1 !important;
    }
    .tab-nav button.selected, .tabs button.selected {
      color: var(--violet) !important;
      background: #f5f3ff !important;
      border-bottom: 3px solid var(--violet) !important;
    }

    .panel {
      background: var(--card);
      border: 1px solid var(--subtle);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
      margin-bottom: 18px;
    }
    .panel h2 { margin: 0 0 10px; font-size: 22px; font-weight: 900; color: var(--ink) !important; }
    .panel h3 { color: var(--ink) !important; }
    .panel.highlight { background: linear-gradient(180deg, #ffffff, #f8fafc); }
    .big-number { font-size: 31px; line-height:1.1; color: var(--blue) !important; font-weight:900; margin: 12px 0; }

    .content-grid { display:grid; grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.85fr); gap:18px; align-items:start; }
    .mini-list div { display:flex; justify-content:space-between; border-top:1px solid var(--subtle); padding:10px 0; }
    .mini-list span { color: var(--muted) !important; }
    .mini-list b { color: var(--ink) !important; }

    .prediction-card {
      border: 1px solid var(--subtle);
      background: white;
      border-radius: 26px;
      padding: 28px;
      box-shadow: var(--shadow);
      min-height: 210px;
    }
    .prediction-card h2 { font-size: 34px; margin: 18px 0 10px; font-weight:900; color: var(--ink) !important; }
    .prediction-card p { color: var(--muted) !important; }
    .prediction-card.sarcastic { border-color:#fecaca; background:linear-gradient(180deg,#fff7ed,#ffffff); }
    .prediction-card.neutral { border-color:#bfdbfe; background:linear-gradient(180deg,#eff6ff,#ffffff); }
    .prediction-card.error { border-color:#fecaca; background:#fff1f2; }
    .badge {
      display:inline-block;
      background:#eef2ff;
      color:#4f46e5 !important;
      border-radius:999px;
      padding:7px 12px;
      text-transform:uppercase;
      font-weight:900;
      letter-spacing:.12em !important;
      font-size:12px;
    }
    .confidence { display:flex; justify-content:space-between; border-top:1px solid var(--subtle); padding-top:16px; margin-top:18px; }
    .confidence b { font-size:24px; color:var(--ink) !important; }
    .quick-meta { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; margin-top:20px; }
    .quick-meta span { border:1px solid var(--subtle); border-radius:16px; padding:12px; background:#f8fafc; color:#334155 !important; font-weight:600; }

    .prob-box { background:white; border:1px solid var(--subtle); border-radius:22px; padding:20px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.04); }
    .prob-box h3 { margin:0 0 18px; color: var(--ink) !important; }
    .prob-row { margin: 14px 0; }
    .prob-row div:first-child, .metric-bar div { display:flex; justify-content:space-between; gap:12px; margin-bottom:8px; }
    .prob-row span, .metric-bar span { color:var(--muted) !important; font-weight:700; }
    .prob-row b, .metric-bar b { color:var(--ink) !important; }
    .bar, .metric-bar em { display:block; height:12px; border-radius:999px; background:#e2e8f0; overflow:hidden; }
    .bar i, .metric-bar i { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--blue),var(--cyan)); border-radius:inherit; }
    .bar.alt i { background:linear-gradient(90deg,var(--violet),#ec4899); }
    .metric-bar { margin: 16px 0; }

    textarea, input, select, .wrap, .block {
      border-radius: 16px !important;
    }
    textarea, input, select {
      color: var(--ink) !important;
      background: #ffffff !important;
      border: 1px solid #cbd5e1 !important;
      font-size: 15px !important;
    }
    textarea::placeholder, input::placeholder { color:#94a3b8 !important; }
    button.primary, .primary {
      background: linear-gradient(90deg, var(--violet), var(--blue), var(--cyan)) !important;
      color: #ffffff !important;
      border: 0 !important;
      font-weight: 900 !important;
      border-radius: 16px !important;
      box-shadow: 0 12px 24px rgba(37,99,235,.18) !important;
    }
    button.secondary { background:#f8fafc !important; color:var(--ink) !important; }

    .table-wrap { overflow:auto; border:1px solid var(--subtle); border-radius:18px; }
    table.clean-table { width:100%; border-collapse:collapse; background:white; font-size:14px; }
    table.clean-table th { background:#f1f5f9; color:#0f172a !important; text-align:left; padding:13px; font-weight:900; border-bottom:1px solid var(--subtle); }
    table.clean-table td { color:#334155 !important; padding:13px; border-bottom:1px solid #eef2f7; vertical-align:top; }
    table.clean-table tr:hover td { background:#f8fafc; }
    .status-done { color:#166534 !important; background:#dcfce7; border-radius:999px; padding:5px 9px; font-weight:900; display:inline-block; }
    .status-warn { color:#92400e !important; background:#fef3c7; border-radius:999px; padding:5px 9px; font-weight:900; display:inline-block; }

    .report-pre, pre, code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
      color:#0f172a !important;
      background:#f8fafc !important;
      border-radius:14px;
      padding:14px;
      white-space:pre-wrap;
    }

    footer { display:none !important; }
    .gradio-container .svelte-1ipelgc { color:var(--ink) !important; }

    @media (max-width: 980px) {
      .hero, .content-grid { grid-template-columns:1fr; }
      .hero { padding:28px; }
      .metric-grid { grid-template-columns:1fr; }
      .quick-meta { grid-template-columns:1fr; }
    }
    """


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate"),
        css=custom_css(),
    ) as demo:
        gr.HTML(hero_html())

        with gr.Tabs():
            with gr.Tab("🏠 Overview"):
                gr.HTML(overview_html())

            with gr.Tab("🔮 Live Prediction"):
                gr.HTML("<div class='panel'><h2>Try the final model</h2><p>Enter a reply comment and optionally add its parent comment. The model returns the class decision, confidence, class probabilities, and combined model input.</p></div>")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        parent = gr.Textbox(label="Optional parent comment / context", lines=4, placeholder="Example: The deadline moved to tomorrow.")
                        comment = gr.Textbox(label="Comment to classify", lines=5, placeholder="Example: Perfect, I love surprise deadlines.")
                        device = gr.Dropdown(["auto", "cpu", "mps", "cuda"], value="auto", label="Inference device")
                        with gr.Row():
                            predict_btn = gr.Button("Predict sarcasm", variant="primary")
                            clear_btn = gr.ClearButton([parent, comment], value="Clear")
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
                        pred_html = gr.HTML(waiting_card())
                        prob_html = gr.HTML(probability_bars())
                        with gr.Accordion("Combined model input", open=False):
                            combined_text = gr.Textbox(label="Text sent to model", lines=4, interactive=False)
                        with gr.Accordion("Technical details", open=False):
                            technical_text = gr.Textbox(label="Compact technical output", lines=7, interactive=False)
                predict_btn.click(predict, inputs=[parent, comment, device], outputs=[pred_html, prob_html, combined_text, technical_text])

            with gr.Tab("📊 Results"):
                # Results tab is intentionally HTML-only to avoid the previous matplotlib/DataFrame rendering crash.
                results_panel = gr.HTML(results_html())
                refresh_results = gr.Button("Refresh results", variant="primary")
                refresh_results.click(results_html, outputs=results_panel)

            with gr.Tab("🧭 Workflow"):
                workflow_panel = gr.HTML(workflow_html())
                refresh_workflow = gr.Button("Refresh workflow status", variant="primary")
                refresh_workflow.click(workflow_html, outputs=workflow_panel)

            with gr.Tab("📚 Reports"):
                gr.HTML("<div class='panel'><h2>Reports explorer</h2><p>Browse generated Markdown, JSON, CSV, and text reports without leaving the dashboard.</p></div>")
                reports = report_files()
                report_dropdown = gr.Dropdown(choices=reports, value=reports[0] if reports else None, label="Report file")
                with gr.Row():
                    open_report = gr.Button("Open report", variant="primary")
                    refresh_report_list = gr.Button("Refresh file list")
                report_output = gr.Markdown("Select a report and click **Open report**.")
                open_report.click(load_report, inputs=report_dropdown, outputs=report_output)
                refresh_report_list.click(lambda: gr.Dropdown(choices=report_files(), value=report_files()[0] if report_files() else None), outputs=report_dropdown)

            with gr.Tab("⚙️ Run Tasks"):
                gr.HTML("<div class='panel'><h2>Local pipeline runner</h2><p>This tab is for local use only. Heavy tasks such as training can take time. Disable task execution for hosted demos using <code>GRADIO_ALLOW_TASK_RUNS=0</code>.</p></div>")
                choices = runner_choices()
                task_select = gr.Dropdown(choices=choices, value=choices[0] if choices else None, label="Task runner")
                confirm = gr.Checkbox(label="I understand this may run a heavy local task", value=False)
                quick = gr.Checkbox(label="Quick mode for supported heavy tasks", value=True)
                run_btn = gr.Button("Run selected task", variant="primary")
                run_log = gr.Textbox(label="Task log", lines=18, interactive=False)
                run_btn.click(run_task, inputs=[task_select, confirm, quick], outputs=run_log)

            with gr.Tab("🛡️ Ethics & Hosting"):
                gr.HTML(ethics_html())

    return demo


demo = build_app()


def main() -> None:
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    main()
