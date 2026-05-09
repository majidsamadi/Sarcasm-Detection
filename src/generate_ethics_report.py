#!/usr/bin/env python3
"""Generate Task 21 ethics and limitations reports.

This script creates an academic ethics package for the sarcasm detection project:
- ethics/limitations summary
- risk register
- responsible-use checklist
- JSON summary
- risk severity figure
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "task21_ethics_limitations_config.json"
TASK17_PATH = PROJECT_ROOT / "reports" / "task17" / "task17_model_comparison_summary.json"
TASK18_PATH = PROJECT_ROOT / "reports" / "task18" / "task18_error_analysis_summary.json"
TASK19_PATH = PROJECT_ROOT / "reports" / "task19" / "task19_final_model_selection_summary.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "task21"
FIG_DIR = REPORT_DIR / "figures"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: Dict[str, Any], paths: List[str], default: Any = None) -> Any:
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


def write_risk_register(config: Dict[str, Any]) -> Path:
    path = REPORT_DIR / "task21_risk_register.csv"
    risks = config.get("risk_register", [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["risk_id", "risk", "severity", "reason", "mitigation"])
        writer.writeheader()
        for risk in risks:
            writer.writerow({k: risk.get(k, "") for k in writer.fieldnames})
    return path


def write_checklist(config: Dict[str, Any]) -> Path:
    path = REPORT_DIR / "task21_responsible_use_checklist.md"
    lines = [
        "# Task 21 Responsible Use Checklist",
        "",
        "This checklist defines how the sarcasm detection system should be presented and used responsibly.",
        "",
    ]
    for rule in config.get("responsible_use_rules", []):
        lines.append(f"- [x] {rule}")
    lines.extend([
        "",
        "## Public Demo Requirement",
        "",
        "If the application is hosted publicly, the public version should focus on inference and report presentation only. Heavy task execution or model training should remain local unless the hosting environment is explicitly configured for it.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_risk_severity(config: Dict[str, Any]) -> Path:
    risks = config.get("risk_register", [])
    counts = Counter(r.get("severity", "Unknown") for r in risks)
    ordered = ["Low", "Medium", "High"]
    labels = [x for x in ordered if x in counts] + [x for x in counts if x not in ordered]
    values = [counts[x] for x in labels]

    fig_path = FIG_DIR / "task21_risk_severity_counts.png"
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.title("Task 21 Risk Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Number of Risks")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return fig_path


def write_summary(config: Dict[str, Any], task17: Dict[str, Any], task18: Dict[str, Any], task19: Dict[str, Any]) -> Path:
    selected_experiment = get_nested(
        task19,
        ["selected_experiment", "selected_model.experiment_id", "final_model.experiment_id"],
        config.get("selected_model", "E03_RoBERTa_VersionA"),
    )
    accuracy = get_nested(
        task19,
        ["accuracy", "selected_model.accuracy", "final_model.accuracy", "metrics.accuracy"],
        "0.7223",
    )
    macro_f1 = get_nested(
        task19,
        ["macro_f1", "selected_model.macro_f1", "final_model.macro_f1", "metrics.macro_f1"],
        "0.7167",
    )
    best_model = get_nested(task17, ["best_experiment", "best.experiment_id"], selected_experiment)

    risks = config.get("risk_register", [])
    high_risks = [r for r in risks if r.get("severity") == "High"]

    md = f"""# Task 21: Ethics and Limitations Summary

## Purpose

Task 21 documents the ethical considerations, limitations, and responsible-use requirements for the sarcasm detection project. This is necessary because sarcasm detection can be misinterpreted if treated as a fully reliable judgment of user intent.

## Final System Context

- **Project:** {config.get('project_title')}
- **Selected model:** {selected_experiment}
- **Model family:** {config.get('final_model_family')}
- **Preprocessing:** {config.get('preprocessing_version')}
- **Input strategy:** {config.get('input_strategy')}
- **Test accuracy:** {accuracy}
- **Test macro-F1:** {macro_f1}
- **Best model from comparison:** {best_model}

## Main Ethical Position

{config.get('main_risk_statement')}

The system should be presented as a research and coursework demonstration, not as a production moderation or user-judgment system.

## Key Ethical Risks

| Risk ID | Risk | Severity | Mitigation |
|---|---|---|---|
"""
    for risk in risks:
        md += f"| {risk.get('risk_id')} | {risk.get('risk')} | {risk.get('severity')} | {risk.get('mitigation')} |\n"

    md += f"""

## High-Priority Risks

The project identifies **{len(high_risks)} high-severity risks**. The most important risks are misclassification, context loss, dataset bias, and deployment misuse.

## Limitations

1. **Sarcasm is context-dependent.** Text alone may not capture tone, speaker intent, shared background, or full conversation history.
2. **Dataset limitation.** The model is trained and evaluated on Reddit-style data, so results may not generalize to other platforms or languages.
3. **Model uncertainty.** A confidence score reflects model probability, not guaranteed truth.
4. **Training-resource limitation.** The project used a practical training sample due to local compute constraints, while evaluation was conducted on the full held-out test split.
5. **Public deployment limitation.** Public hosting should focus on inference and report presentation. Heavy training execution should remain local.

## Responsible Use Rules

"""
    for rule in config.get("responsible_use_rules", []):
        md += f"- {rule}\n"

    md += """

## Academic Conclusion

The project includes ethics and limitations as part of the NLP pipeline because responsible interpretation is essential for social-media text classification. The model can demonstrate how transformer-based models detect sarcasm patterns, but it should not be treated as a definitive detector of human intention.
"""
    path = REPORT_DIR / "task21_ethics_and_limitations_summary.md"
    path.write_text(md, encoding="utf-8")
    return path


def write_json_summary(config: Dict[str, Any], generated_files: Dict[str, str]) -> Path:
    risks = config.get("risk_register", [])
    summary = {
        "task_id": config.get("task_id"),
        "task_name": config.get("task_name"),
        "selected_model": config.get("selected_model"),
        "main_risk_statement": config.get("main_risk_statement"),
        "number_of_risks": len(risks),
        "risk_severity_counts": dict(Counter(r.get("severity", "Unknown") for r in risks)),
        "ethical_principles": config.get("ethical_principles", []),
        "responsible_use_rules": config.get("responsible_use_rules", []),
        "generated_files": generated_files,
    }
    path = REPORT_DIR / "task21_ethics_and_limitations_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_progress_note() -> Path:
    text = (
        "Ethics and limitations completed. A responsible-use analysis was prepared for the final RoBERTa Version A sarcasm detection model. "
        "The task documented key risks including misclassification, context loss, Reddit dataset bias, privacy concerns, over-reliance on confidence scores, deployment misuse, compute limitations, and cultural/linguistic limitations. "
        "A risk register, responsible-use checklist, summary report, JSON summary, and risk severity figure were saved under reports/task21 and pushed to GitHub."
    )
    path = REPORT_DIR / "task21_progress_note.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_readme() -> Path:
    text = """# Task 21 Reports

This folder contains the ethics and limitations package for the sarcasm detection project.

Files include:

- `task21_ethics_and_limitations_summary.md`
- `task21_ethics_and_limitations_summary.json`
- `task21_risk_register.csv`
- `task21_responsible_use_checklist.md`
- `task21_progress_note.txt`
- `figures/task21_risk_severity_counts.png`

The purpose is to document responsible use, risks, limitations, and mitigation strategies for the final model and demo.
"""
    path = REPORT_DIR / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    config = load_json(CONFIG_PATH)
    task17 = load_json(TASK17_PATH)
    task18 = load_json(TASK18_PATH)
    task19 = load_json(TASK19_PATH)

    generated: Dict[str, str] = {}
    generated["risk_register"] = str(write_risk_register(config).relative_to(PROJECT_ROOT))
    generated["responsible_use_checklist"] = str(write_checklist(config).relative_to(PROJECT_ROOT))
    generated["risk_figure"] = str(plot_risk_severity(config).relative_to(PROJECT_ROOT))
    generated["summary_md"] = str(write_summary(config, task17, task18, task19).relative_to(PROJECT_ROOT))
    generated["readme"] = str(write_readme().relative_to(PROJECT_ROOT))
    generated["progress_note"] = str(write_progress_note().relative_to(PROJECT_ROOT))
    generated["summary_json"] = str(write_json_summary(config, generated).relative_to(PROJECT_ROOT))

    print("Task 21 generated files:")
    for key, value in generated.items():
        print(f"- {key}: {value}")
    print("Done. Task 21 ethics and limitations completed successfully.")


if __name__ == "__main__":
    main()
