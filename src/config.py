from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RANDOM_STATE = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
