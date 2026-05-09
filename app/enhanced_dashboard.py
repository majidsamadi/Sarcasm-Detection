#!/usr/bin/env python3
"""Enhanced Streamlit dashboard for the full AI/NLP workflow.

This dashboard is intended to show the full academic pipeline:
preprocessing, splitting, experiment design, training, evaluation, analysis,
final model selection, and demo inference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_sarcasm import SarcasmPredictor  # noqa: E402
from src.ui_report_loader import (  # noqa: E402
    TASKS,
    all_task_statuses,
    exists,
    find_report_files,
    load_final_model_summary,
    load_task15_metrics,
    load_task17_table,
    overall_progress,
    read_json,
    read_text,
    simple_markdown_preview,
)
from src.ui_task_runner import command_for_runner, latest_logs, stream_command  # noqa: E402


st.set_page_config(
    page_title="Sarcasm Detection NLP Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
  --card-radius: 18px;
}
.main .block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}
.hero-card {
  background: linear-gradient(135deg, #1f2937 0%, #312e81 45%, #7c2d12 100%);
  color: white;
  padding: 28px 32px;
  border-radius: 24px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.20);
  margin-bottom: 20px;
}
.hero-card h1 {
  margin: 0;
  font-size: 2.15rem;
  line-height: 1.15;
}
.hero-card p {
  margin: 8px 0 0 0;
  opacity: 0.92;
  font-size: 1.02rem;
}
.metric-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: var(--card-radius);
  padding: 18px 20px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
  min-height: 112px;
}
.metric-card .label {
  color: #6b7280;
  font-size: 0.88rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.metric-card .value {
  color: #111827;
  font-size: 1.9rem;
  font-weight: 800;
  margin-top: 6px;
}
.metric-card .sub {
  color: #6b7280;
  font-size: 0.9rem;
  margin-top: 4px;
}
.status-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.78rem;
}
.done-pill { background: #dcfce7; color: #166534; }
.pending-pill { background: #fee2e2; color: #991b1b; }
.local-pill { background: #fef3c7; color: #92400e; }
.report-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 16px;
}
.small-muted { color: #64748b; font-size: 0.88rem; }
.warning-box {
  background: #fff7ed;
  border: 1px solid #fed7aa;
  color: #9a3412;
  border-radius: 16px;
  padding: 14px 16px;
}
.success-box {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  color: #166534;
  border-radius: 16px;
  padding: 14px 16px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading selected RoBERTa model...")
def load_predictor() -> SarcasmPredictor:
    return SarcasmPredictor(config_path=PROJECT_ROOT / "configs" / "final_model_config.json")


def card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(done: bool) -> str:
    if done:
        return '<span class="status-pill done-pill">Done</span>'
    return '<span class="status-pill pending-pill">Pending</span>'


def local_pill() -> str:
    return '<span class="status-pill local-pill">Local only</span>'


def render_header() -> None:
    progress = overall_progress()
    st.markdown(
        f"""
        <div class="hero-card">
          <h1>🧠 Sarcasm Detection NLP Lab</h1>
          <p>Interactive dashboard for the full AI/NLP workflow: preprocessing → training → evaluation → analysis → final demo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Pipeline Tasks", f"{progress['completed']}/{progress['total']}", "Completed workflow steps")
    with c2:
        card("Final Model", "RoBERTa A", "Stopwords kept")
    with c3:
        card("Test Accuracy", "0.7223", "Held-out test split")
    with c4:
        card("Test Macro-F1", "0.7167", "Primary selection metric")
    st.progress(float(progress["pct"]), text=f"Overall pipeline completion: {progress['pct']*100:.1f}%")


def page_overview() -> None:
    render_header()
    st.subheader("Academic Workflow Status")
    df = all_task_statuses()
    display = df.copy()
    display["status"] = display["completed"].map(lambda x: "Done" if x else "Pending")
    display["scope"] = display.apply(lambda r: "Local heavy task" if r["local_only"] else "Safe report/UI task", axis=1)
    display = display[["number", "title", "category", "status", "checks_passed", "checks_total", "scope"]]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Experiment Results Snapshot")
    metrics = load_task15_metrics()
    if metrics.empty:
        st.info("Task 15 metrics are not available yet.")
    else:
        cols = ["experiment", "model", "preprocessing", "accuracy", "macro_f1", "weighted_f1"]
        st.dataframe(metrics[cols], use_container_width=True, hide_index=True)
        chart_df = metrics.set_index("experiment")[["accuracy", "macro_f1"]]
        st.bar_chart(chart_df)

    st.subheader("Best Model Rationale")
    st.markdown(
        """
        <div class="success-box">
        <b>Selected final model:</b> RoBERTa Version A. It achieved the best held-out test Macro-F1 and uses the preprocessing version where stopwords are kept, which performed better for sarcasm detection.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_pipeline_runner() -> None:
    st.title("⚙️ Pipeline Runner")
    st.markdown(
        """
        <div class="warning-box">
        <b>Local-use recommendation:</b> Run heavy training and evaluation tasks from your local machine only. On Hugging Face Spaces, use this page as read-only or disable task execution because compute and file persistence are limited.
        </div>
        """,
        unsafe_allow_html=True,
    )

    task_options = {f"Task {t.number}: {t.title}": t for t in TASKS}
    selected_label = st.selectbox("Select workflow task", list(task_options.keys()), index=len(task_options)-1)
    task = task_options[selected_label]

    status = all_task_statuses()
    row = status[status["number"] == task.number].iloc[0].to_dict()

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Status:** {'✅ Done' if row['completed'] else '⏳ Pending'}")
    c2.markdown(f"**Category:** {task.category}")
    c3.markdown(f"**Execution:** {'Heavy/local' if task.local_only else 'Light/report'}")

    if task.primary_report and exists(task.primary_report):
        with st.expander("View current report", expanded=False):
            st.markdown(simple_markdown_preview(read_text(task.primary_report)))

    if not task.runner_sh:
        st.info("No runner script is registered for this task.")
        return

    st.markdown("### Run settings")
    mode = st.radio(
        "Execution mode",
        ["Do not run", "Run task"],
        horizontal=True,
        help="Choose 'Run task' only when you are ready. Heavy tasks may take time.",
    )

    env_overrides: Dict[str, str] = {}
    if task.number in {13, 14, 15}:
        run_mode = st.selectbox(
            "Training/evaluation size mode",
            ["default practical", "small smoke", "full data"],
            help="Small smoke is useful for checking code. Default practical uses the existing script settings. Full data may take much longer.",
        )
        if task.number == 13:
            if run_mode == "small smoke":
                env_overrides.update({"TASK13_MAX_TRAIN_SAMPLES": "32", "TASK13_MAX_VALID_SAMPLES": "32", "TASK13_OVERWRITE": "1"})
            elif run_mode == "full data":
                env_overrides.update({"TASK13_FULL_DATA": "1", "TASK13_OVERWRITE": "1"})
        if task.number == 14:
            if run_mode == "small smoke":
                env_overrides.update({"TASK14_MAX_TRAIN_SAMPLES": "32", "TASK14_MAX_VALID_SAMPLES": "32", "TASK14_OVERWRITE": "1"})
            elif run_mode == "full data":
                env_overrides.update({"TASK14_FULL_DATA": "1", "TASK14_OVERWRITE": "1"})
        if task.number == 15:
            if run_mode == "small smoke":
                env_overrides.update({"TASK15_MAX_TEST_SAMPLES": "1000"})
            elif run_mode == "full data":
                env_overrides.update({"TASK15_MAX_TEST_SAMPLES": ""})

    if mode == "Run task":
        confirm = st.checkbox("I understand this will run a local script and may take time.")
        if st.button(f"▶ Run {selected_label}", type="primary", disabled=not confirm):
            try:
                command = command_for_runner(task.runner_sh)
                st.write("Command:", " ".join(command))
                log_area = st.empty()
                progress = st.progress(0, text="Starting task...")
                lines: List[str] = []
                count = 0
                finished_line = ""
                for line in stream_command(command, log_name=f"task{task.number:02d}_{task.title}", env_overrides=env_overrides):
                    count += 1
                    if line.startswith("__TASK_FINISHED__"):
                        finished_line = line
                        progress.progress(1.0, text="Task finished")
                    else:
                        lines.append(line)
                        if len(lines) > 120:
                            lines = lines[-120:]
                        progress.progress(min(0.95, count / 500), text=f"Running... streamed {count} lines")
                        log_area.code("\n".join(lines), language="bash")
                st.success(finished_line or "Task completed.")
                st.rerun()
            except Exception as exc:
                st.error("Task execution failed.")
                st.exception(exc)

    st.markdown("### Recent UI run logs")
    logs = latest_logs()
    if not logs:
        st.caption("No UI run logs yet.")
    else:
        for log_path in logs[:6]:
            rel = log_path.relative_to(PROJECT_ROOT)
            with st.expander(str(rel)):
                st.code(log_path.read_text(encoding="utf-8", errors="replace")[-12000:], language="bash")


def page_model_demo() -> None:
    st.title("🔮 Live Sarcasm Prediction")
    st.caption("Uses selected final model: RoBERTa Version A")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        parent_comment = st.text_area("Optional parent comment / context", height=110, placeholder="The deadline moved to tomorrow.")
        comment = st.text_area("Comment to classify", height=140, placeholder="Perfect, I love surprise deadlines.")
        predict = st.button("Predict Sarcasm", type="primary", use_container_width=True)
    with c2:
        st.markdown("### Model Card")
        model_card = read_text("reports/task19/final_model_card.md", "Model card not available.")
        st.markdown(simple_markdown_preview(model_card, max_chars=4000))

    if predict:
        try:
            predictor = load_predictor()
            result = predictor.predict(comment=comment, parent_comment=parent_comment)
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction", result.label)
            r2.metric("Confidence", f"{result.confidence:.4f}")
            r3.metric("Checkpoint", "RoBERTa A")
            prob_df = pd.DataFrame({
                "class": ["Non-sarcastic", "Sarcastic"],
                "probability": [result.probability_non_sarcastic, result.probability_sarcastic],
            }).set_index("class")
            st.bar_chart(prob_df)
            with st.expander("Technical output"):
                st.json(result.to_dict())
        except Exception as exc:
            st.error("Prediction failed. Check local model checkpoint or Hugging Face model configuration.")
            st.exception(exc)


def page_reports() -> None:
    st.title("📚 Reports Explorer")
    rows = find_report_files()
    if not rows:
        st.info("No reports found.")
        return
    df = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 2])
    with c1:
        file_type = st.multiselect("File type", sorted(df["type"].unique()), default=sorted(df["type"].unique()))
    filtered = df[df["type"].isin(file_type)]
    selected = st.selectbox("Select report/document", filtered["file"].tolist())
    path = PROJECT_ROOT / selected
    st.markdown(f"**Selected:** `{selected}`")
    if path.suffix.lower() == ".json":
        st.json(read_json(selected))
    elif path.suffix.lower() == ".csv":
        try:
            st.dataframe(pd.read_csv(path), use_container_width=True)
        except Exception:
            st.code(path.read_text(encoding="utf-8", errors="replace"))
    else:
        st.markdown(simple_markdown_preview(path.read_text(encoding="utf-8", errors="replace")))


def page_results() -> None:
    st.title("📊 Results Dashboard")
    metrics = load_task15_metrics()
    if metrics.empty:
        st.info("Task 15 metrics are missing.")
        return
    st.subheader("All Experiments")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Accuracy vs Macro-F1")
    chart = metrics.set_index("experiment")[["accuracy", "macro_f1"]]
    st.bar_chart(chart)

    st.subheader("Stopword Impact")
    keep = metrics[metrics["preprocessing"] == "Stopwords kept"]
    remove = metrics[metrics["preprocessing"] == "Selective stopword removal"]
    if not keep.empty and not remove.empty:
        pivot = metrics.pivot_table(index="model", columns="preprocessing", values="macro_f1")
        st.dataframe(pivot, use_container_width=True)
        st.bar_chart(pivot)
        st.markdown(
            """
            <div class="success-box">
            <b>Conclusion:</b> Keeping stopwords performed better for both BERTweet and RoBERTa. This supports the interpretation that sarcasm relies on sentence structure, context, and function words.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Task 17 Ranking")
    task17 = load_task17_table()
    if not task17.empty:
        st.dataframe(task17, use_container_width=True, hide_index=True)


def page_error_analysis() -> None:
    st.title("🧪 Error Analysis")
    summary = read_text("reports/task18/task18_error_analysis_summary.md", "Task 18 summary not found.")
    st.markdown(simple_markdown_preview(summary))
    st.divider()
    st.subheader("Classification Report")
    report = read_text("reports/task18/task18_classification_report.txt", "Classification report not found.")
    st.code(report)
    st.subheader("Available Figures")
    fig_dir = PROJECT_ROOT / "reports" / "task18" / "figures"
    if fig_dir.exists():
        figs = sorted([p for p in fig_dir.glob("*.png")])
        if figs:
            cols = st.columns(2)
            for idx, fig in enumerate(figs):
                with cols[idx % 2]:
                    st.image(str(fig), caption=fig.name, use_container_width=True)
        else:
            st.info("No Task 18 figures found.")


def page_hosting() -> None:
    st.title("🚀 Hugging Face Hosting Readiness")
    st.markdown(
        """
        This page explains the recommended deployment architecture.

        **Recommended setup:**
        - GitHub: source code, reports, scripts, documentation
        - Hugging Face Model Hub: final RoBERTa Version A checkpoint
        - Hugging Face Space: Streamlit or Docker-hosted interactive app
        """
    )
    st.markdown(
        """
        <div class="warning-box">
        Training and pipeline execution should be disabled in the public Hugging Face Space unless a suitable persistent storage and compute plan is configured. The hosted app should focus on demo inference and report viewing.
        </div>
        """,
        unsafe_allow_html=True,
    )
    checks = {
        "Final model config": exists("configs/final_model_config.json"),
        "Final model card": exists("reports/task19/final_model_card.md"),
        "Local checkpoint": (PROJECT_ROOT / "models" / "roberta" / "versionA").exists(),
        "Enhanced dashboard": exists("app/enhanced_dashboard.py"),
        "Prediction utility": exists("src/predict_sarcasm.py"),
    }
    for name, ok in checks.items():
        st.write(("✅" if ok else "❌") + f" {name}")

    st.subheader("Next hosting tasks")
    st.markdown(
        """
        1. Upload `models/roberta/versionA` to a Hugging Face model repository.
        2. Modify the app to load the model from the Hugging Face model repository.
        3. Create a Hugging Face Space.
        4. Disable heavy task execution on hosted version.
        5. Test inference URL and share it with the lecturer/team.
        """
    )


def main() -> None:
    with st.sidebar:
        st.markdown("## 🧠 NLP Lab")
        st.caption("Sarcasm Detection Workflow")
        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Pipeline Runner",
                "Live Demo",
                "Results Dashboard",
                "Reports Explorer",
                "Error Analysis",
                "Hosting Readiness",
            ],
        )
        st.divider()
        progress = overall_progress()
        st.metric("Completed Tasks", f"{progress['completed']}/{progress['total']}")
        st.progress(float(progress["pct"]))
        st.caption("Local enhanced dashboard. Public hosting should use inference/report mode only.")

    if page == "Overview":
        page_overview()
    elif page == "Pipeline Runner":
        page_pipeline_runner()
    elif page == "Live Demo":
        page_model_demo()
    elif page == "Results Dashboard":
        page_results()
    elif page == "Reports Explorer":
        page_reports()
    elif page == "Error Analysis":
        page_error_analysis()
    elif page == "Hosting Readiness":
        page_hosting()


if __name__ == "__main__":
    main()
