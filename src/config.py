from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

VERSION_A_FILE = PROCESSED_DIR / "A.csv"
VERSION_B_FILE = PROCESSED_DIR / "B.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

LABEL_CANDIDATES = ["label", "is_sarcastic", "sarcasm", "target", "class"]
TEXT_CANDIDATES = ["comment", "text", "body"]
CONTEXT_CANDIDATES = ["parent_comment", "context", "parent"]
