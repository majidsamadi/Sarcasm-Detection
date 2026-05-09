from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "task12_experiment_design.json"
REPORT_MD = PROJECT_ROOT / "reports" / "task12_experiment_design_summary.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "task12_experiment_design_summary.json"
PROGRESS_NOTE = PROJECT_ROOT / "reports" / "task12_progress_note.txt"

REQUIRED_COLUMNS = {"label", "comment", "parent_comment"}
SPLIT_FILES = ["train.csv", "valid.csv", "test.csv"]


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required split file: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def label_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["label"].value_counts().sort_index().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def validate_experiment_registry(config: Dict) -> List[str]:
    messages: List[str] = []
    experiments = config.get("experiments", [])

    if len(experiments) != 4:
        raise ValueError(f"Expected exactly 4 experiments, found {len(experiments)}")

    experiment_ids = [exp["experiment_id"] for exp in experiments]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Experiment IDs must be unique.")

    expected = {
        ("BERTweet", "versionA"),
        ("BERTweet", "versionB"),
        ("RoBERTa", "versionA"),
        ("RoBERTa", "versionB"),
    }
    actual = {(exp["model_family"], exp["data_version"]) for exp in experiments}
    if actual != expected:
        raise ValueError(f"Experiment matrix is not correct. Expected {expected}, got {actual}")

    messages.append("Experiment registry contains exactly four required experiments.")
    messages.append("Experiment matrix covers BERTweet and RoBERTa across Version A and Version B.")
    return messages


def validate_splits(config: Dict) -> Tuple[List[str], Dict]:
    messages: List[str] = []
    split_summary: Dict = {}

    data_versions = config["data_versions"]
    for version_key, version_info in data_versions.items():
        folder = PROJECT_ROOT / version_info["split_folder"]
        split_summary[version_key] = {}
        for split_file in SPLIT_FILES:
            split_name = split_file.replace(".csv", "")
            path = folder / split_file
            df = read_split(path)
            split_summary[version_key][split_name] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "rows": int(len(df)),
                "label_counts": label_counts(df),
                "columns": list(df.columns),
                "empty_comment_rows": int(df["comment"].isna().sum() + (df["comment"].astype(str).str.strip() == "").sum()),
                "empty_parent_comment_rows": int(df["parent_comment"].isna().sum() + (df["parent_comment"].astype(str).str.strip() == "").sum()),
            }
        messages.append(f"Validated split files for {version_key}.")

    # Fairness check: Version A and Version B should have matching row counts and label counts per split.
    for split_name in ["train", "valid", "test"]:
        a = split_summary["versionA"][split_name]
        b = split_summary["versionB"][split_name]
        if a["rows"] != b["rows"]:
            raise ValueError(f"Row count mismatch for {split_name}: A={a['rows']} B={b['rows']}")
        if a["label_counts"] != b["label_counts"]:
            raise ValueError(f"Label distribution mismatch for {split_name}: A={a['label_counts']} B={b['label_counts']}")
    messages.append("Version A and Version B have matching row counts and label distributions for all splits.")

    return messages, split_summary


def write_reports(config: Dict, validation_messages: List[str], split_summary: Dict) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    experiments = config["experiments"]
    training = config["training_design"]
    tokenization = config["tokenization_design"]
    metrics = config["metrics"]

    lines: List[str] = []
    lines.append("# Task 12: Model Experiment Design Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append(config["objective"])
    lines.append("")
    lines.append("## Task Definition")
    lines.append("- Task type: Binary sarcasm classification")
    lines.append("- Positive class: 1 = sarcastic")
    lines.append("- Negative class: 0 = non-sarcastic")
    lines.append("- Input strategy: text-only context pair using `parent_comment` and `comment`")
    lines.append("- No user metadata is used")
    lines.append("")
    lines.append("## Experiment Matrix")
    lines.append("| Experiment ID | Model | Hugging Face Model | Data Version | Stopword Setting |")
    lines.append("|---|---|---|---|---|")
    for exp in experiments:
        lines.append(
            f"| {exp['experiment_id']} | {exp['model_family']} | "
            f"{exp['huggingface_model']} | {exp['data_version']} | {exp['stopword_setting']} |"
        )
    lines.append("")
    lines.append("## Controlled Variables")
    lines.append(f"- Random seed: {config['split_design']['random_seed']}")
    lines.append(f"- Split design: stratified 80/10/10 split")
    lines.append(f"- Max sequence length: {tokenization['max_length']}")
    lines.append(f"- Padding: {tokenization['padding']}")
    lines.append(f"- Truncation: {tokenization['truncation']}")
    lines.append(f"- Epochs: {training['epochs']}")
    lines.append(f"- Batch size: {training['batch_size']}")
    lines.append(f"- Learning rate: {training['learning_rate']}")
    lines.append(f"- Weight decay: {training['weight_decay']}")
    lines.append("")
    lines.append("## Evaluation Metrics")
    for metric in metrics:
        lines.append(f"- {metric}")
    lines.append("")
    lines.append("## Split Validation Summary")
    lines.append("| Version | Split | Rows | Label 0 | Label 1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for version_key, splits in split_summary.items():
        for split_name, info in splits.items():
            counts = info["label_counts"]
            lines.append(
                f"| {version_key} | {split_name} | {info['rows']} | "
                f"{counts.get('0', 0)} | {counts.get('1', 0)} |"
            )
    lines.append("")
    lines.append("## Validation Checks")
    for msg in validation_messages:
        lines.append(f"- PASS: {msg}")
    lines.append("")
    lines.append("## Model Selection Rule")
    lines.append(
        "The final model will be selected primarily based on F1-score, then checked using precision, "
        "recall, confusion matrix, and error analysis. Only one final model will be deployed in the demo."
    )
    lines.append("")
    lines.append("## Academic Rationale")
    rationale = config["academic_rationale"]
    lines.append(f"- Why two models: {rationale['why_two_models']}")
    lines.append(f"- Why two preprocessing versions: {rationale['why_two_preprocessing_versions']}")
    lines.append(f"- Why one final model: {rationale['why_one_final_model']}")
    lines.append("")
    lines.append("## Status")
    lines.append("Task 12 is complete. The experiment design is ready for Task 13 and Task 14 model training.")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = {
        "task_id": "12",
        "status": "complete",
        "validation_messages": validation_messages,
        "split_summary": split_summary,
        "experiments": experiments,
        "model_selection_rule": config["model_selection_rule"],
    }
    REPORT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    progress = (
        "Model experiment design completed. Four controlled experiments were defined: "
        "BERTweet with Version A, BERTweet with Version B, RoBERTa with Version A, and RoBERTa with Version B. "
        "All experiments use the same stratified train/validation/test split, text-only input, max length 128, "
        "and the same evaluation metrics: accuracy, precision, recall, F1-score, and confusion matrix. "
        "The final implementation will deploy only the best-performing model based mainly on F1-score and error analysis."
    )
    PROGRESS_NOTE.write_text(progress + "\n", encoding="utf-8")


def main() -> None:
    print("Task 12: Model Experiment Design")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config: {CONFIG_PATH}")

    config = load_config()
    messages = []
    messages.extend(validate_experiment_registry(config))
    split_messages, split_summary = validate_splits(config)
    messages.extend(split_messages)
    write_reports(config, messages, split_summary)

    print("\nValidation messages:")
    for msg in messages:
        print(f"PASS: {msg}")
    print(f"\nSaved report: {REPORT_MD}")
    print(f"Saved JSON summary: {REPORT_JSON}")
    print(f"Saved progress note: {PROGRESS_NOTE}")
    print("\nDone. Task 12 completed successfully.")


if __name__ == "__main__":
    main()
