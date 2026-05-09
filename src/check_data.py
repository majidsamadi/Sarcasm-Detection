"""
Quick helper to check whether A.csv and B.csv are available locally.
"""

from pathlib import Path
import pandas as pd

from config import VERSION_A_FILE, VERSION_B_FILE


def check_file(path: Path) -> None:
    if not path.exists():
        print(f"Missing: {path}")
        return
    df = pd.read_csv(path, nrows=5)
    print(f"Found: {path}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    print()


if __name__ == "__main__":
    check_file(VERSION_A_FILE)
    check_file(VERSION_B_FILE)
