"""
Phense Data Utilities
Helpers for loading, cleaning, and encoding the training dataset.
Used by train.py and the exploration notebooks.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── Expected columns ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rainfall_mm",
    "elevation_m",
    "river_distance_km",
    "soil_type",
    "drainage",
]
TARGET_COL = "risk_class"

CATEGORICAL_COLS = ["soil_type", "drainage"]
NUMERIC_COLS     = ["rainfall_mm", "elevation_m", "river_distance_km"]

VALID_SOIL_TYPES = {"clay", "sandy", "loamy", "silty"}
VALID_DRAINAGE   = {"none", "basic", "moderate", "good"}
VALID_RISK       = {"Low", "Medium", "High", "Critical"}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_raw(filename: str = "nigeria_flood_training.csv") -> pd.DataFrame:
    """
    Load the primary training CSV from data/processed/.
    Raises FileNotFoundError with a helpful message if it doesn't exist yet.
    """
    path = os.path.join(_ROOT, "data", "processed", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training data not found at {path}.\n"
            "Steps to create it:\n"
            "  1. Download raw datasets from NiMet / NIHSA / HDX (see README.md).\n"
            "  2. Place them in data/raw/.\n"
            "  3. Run notebooks/01_data_exploration.ipynb to inspect and clean.\n"
            "  4. The cleaned file will be saved to data/processed/ automatically."
        )
    return pd.read_csv(path)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assert required columns exist and values are within expected ranges.
    Returns the dataframe if valid; raises ValueError with a clear report if not.
    """
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    errors = []

    # Categorical value checks
    bad_soil  = ~df["soil_type"].str.lower().isin(VALID_SOIL_TYPES)
    bad_drain = ~df["drainage"].str.lower().isin(VALID_DRAINAGE)
    bad_risk  = ~df[TARGET_COL].isin(VALID_RISK)
    if bad_soil.any():
        errors.append(f"Invalid soil_type values in {bad_soil.sum()} rows. Expected: {VALID_SOIL_TYPES}")
    if bad_drain.any():
        errors.append(f"Invalid drainage values in {bad_drain.sum()} rows. Expected: {VALID_DRAINAGE}")
    if bad_risk.any():
        errors.append(f"Invalid risk_class values in {bad_risk.sum()} rows. Expected: {VALID_RISK}")

    # Numeric range checks
    if (df["rainfall_mm"] < 0).any() or (df["rainfall_mm"] > 800).any():
        errors.append("rainfall_mm contains values outside [0, 800].")
    if (df["elevation_m"] < 0).any() or (df["elevation_m"] > 1500).any():
        errors.append("elevation_m contains values outside [0, 1500].")
    if (df["river_distance_km"] < 0).any():
        errors.append("river_distance_km contains negative values.")

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise and clean the raw dataframe.
    Steps:
      1. Strip whitespace from string columns.
      2. Lowercase categorical columns.
      3. Drop duplicate rows.
      4. Drop rows where any required column is null.
      5. Report what was dropped.
    """
    original_len = len(df)

    # Standardise strings
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Restore title-case for risk_class (model expects "Low" not "low")
    df[TARGET_COL] = df[TARGET_COL].str.title()

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows with nulls in required columns
    required = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=required)

    dropped = original_len - len(df)
    if dropped:
        print(f"[data_utils] Dropped {dropped} rows during cleaning ({original_len} → {len(df)}).")

    return df.reset_index(drop=True)


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features as ordered integers.
    Returns the encoded dataframe and a dict of LabelEncoders for saving.

    Encoding order is fixed (not learned) to ensure reproducibility:
      soil_type : clay=0, loamy=1, sandy=2, silty=3
      drainage  : none=0, basic=1, moderate=2, good=3
      risk_class: Low=0, Medium=1, High=2, Critical=3
    """
    df = df.copy()

    SOIL_ORDER  = ["clay", "loamy", "sandy", "silty"]
    DRAIN_ORDER = ["none", "basic", "moderate", "good"]
    RISK_ORDER  = ["Low", "Medium", "High", "Critical"]

    encoders = {}

    for col, order in [("soil_type", SOIL_ORDER), ("drainage", DRAIN_ORDER)]:
        le = LabelEncoder()
        le.classes_ = np.array(order)
        df[col + "_enc"] = le.transform(df[col])
        encoders[col] = le

    le_risk = LabelEncoder()
    le_risk.classes_ = np.array(RISK_ORDER)
    df[TARGET_COL + "_enc"] = le_risk.transform(df[TARGET_COL])
    encoders[TARGET_COL] = le_risk

    return df, encoders


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """Return X (features) and y (target) as numpy arrays, ready for sklearn."""
    X = df[["rainfall_mm", "elevation_m", "river_distance_km",
            "soil_type_enc", "drainage_enc"]].values
    y = df["risk_class_enc"].values
    return X, y
