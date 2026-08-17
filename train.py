"""
Phense Training Pipeline
Run this script once your dataset is ready in data/processed/.

Usage:
    python train.py

Output:
    models/phense_model.pkl   — trained Random Forest classifier
    models/encoders.pkl       — LabelEncoders for soil_type, drainage, risk_class
    models/metrics.json       — accuracy, classification report, feature importances
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data_utils import load_raw, validate, clean, encode_features, get_feature_matrix

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE   = 0.20   # 80 / 20 train-test split


def main():
    print("=" * 60)
    print("  PHENSE — Training Pipeline")
    print("=" * 60)

    # ── Phase 1: Load ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df_raw = load_raw()
    print(f"      Loaded {len(df_raw):,} records.")

    # ── Phase 2: Clean & validate ─────────────────────────────────────────────
    print("\n[2/5] Cleaning and validating...")
    df = clean(df_raw)
    validate(df)
    print(f"      {len(df):,} records after cleaning.")
    print(f"      Class distribution:\n{df['risk_class'].value_counts().to_string()}")

    # ── Phase 3: Encode & split ───────────────────────────────────────────────
    print("\n[3/5] Encoding features and splitting data...")
    df_enc, encoders = encode_features(df)
    X, y = get_feature_matrix(df_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Phase 4: Train ────────────────────────────────────────────────────────
    print("\n[4/5] Training Random Forest classifier...")
    print("      Running GridSearchCV (this may take a few minutes)...")

    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
        "class_weight":      ["balanced"],   # handles class imbalance
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print(f"      Best params: {grid_search.best_params_}")
    print(f"      Best CV accuracy: {grid_search.best_score_:.4f}")

    # ── Phase 5: Evaluate ─────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set...")
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=list(encoders["risk_class"].classes_),
        output_dict=True,
    )
    cm     = confusion_matrix(y_test, y_pred).tolist()

    feature_names = [
        "rainfall_mm", "elevation_m", "river_distance_km",
        "soil_type_enc", "drainage_enc",
    ]
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    top_feature = max(importances, key=importances.get)

    print(f"\n  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Top Feature   : {top_feature} ({importances[top_feature]:.4f})")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(encoders["risk_class"].classes_)))
    print("  Confusion Matrix:")
    print(np.array(cm))

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("\nSaving model artefacts to models/...")

    joblib.dump(model,    os.path.join(MODELS_DIR, "phense_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))

    metrics = {
        "accuracy":           round(acc, 6),
        "best_cv_accuracy":   round(grid_search.best_score_, 6),
        "best_params":        grid_search.best_params_,
        "feature_importances": {k: round(v, 6) for k, v in importances.items()},
        "top_feature":        top_feature,
        "classification_report": report,
        "confusion_matrix":   cm,
        "class_labels":       list(encoders["risk_class"].classes_),
        "n_train":            len(X_train),
        "n_test":             len(X_test),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("  models/phense_model.pkl  [OK]")
    print("  models/encoders.pkl      [OK]")
    print("  models/metrics.json      [OK]")
    print("\n" + "=" * 60)
    print(f"  Training complete. Accuracy: {acc*100:.2f}%")
    print("  Run `python app.py` to start the API server.")
    print("=" * 60)


if __name__ == "__main__":
    main()
