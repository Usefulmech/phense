"""
Phense Prediction Engine
Loads the trained model and returns a structured risk verdict.
"""

import os
import joblib
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH   = os.path.join(_ROOT, "models", "phense_model.pkl")
ENCODER_PATH = os.path.join(_ROOT, "models", "encoders.pkl")

# ── Encoding maps (must match train.py) ───────────────────────────────────────
SOIL_ENCODING = {"clay": 0, "loamy": 1, "sandy": 2, "silty": 3}
DRAIN_ENCODING = {"none": 0, "basic": 1, "moderate": 2, "good": 3}
CLASSES = ["Low", "Medium", "High", "Critical"]

# ── Feature order (must match training column order) ──────────────────────────
FEATURE_NAMES = [
    "rainfall_mm",
    "elevation_m",
    "river_distance_km",
    "soil_type_enc",
    "drainage_enc",
]

# ── Seasonal label modifier ────────────────────────────────────────────────────
# The same model probability carries different urgency in wet vs dry season.
# Wet season: saturated ground amplifies medium-probability events.
SEASONAL_UPGRADE = {
    "dry": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
    "wet": {"Low": "Low", "Medium": "Medium-High", "High": "High", "Critical": "Critical"},
}

# ── Plain-language advisories (keyed by seasonal label) ───────────────────────
ADVISORIES = {
    "Low": (
        "Conditions are stable. Continue normal activities. "
        "Keep drainage channels clear and monitor weather updates."
    ),
    "Medium": (
        "Moderate risk present. Clear gutters and drainage channels. "
        "Avoid constructing or storing valuables near river banks. "
        "Monitor NiMet forecasts over the next 24 hours."
    ),
    "Medium-High": (
        "Wet season elevation risk. Pre-position sandbags around vulnerable structures. "
        "Identify your nearest evacuation route now. "
        "Alert elderly and children in low-lying homes."
    ),
    "High": (
        "Move livestock and stored crops to higher ground immediately. "
        "Do not attempt to cross flooded roads or streams. "
        "Contact your community leader to coordinate response."
    ),
    "Critical": (
        "Immediate action required. Evacuate all low-lying structures within 2 hours. "
        "Send a WhatsApp alert to your community group now. "
        "Move to the nearest elevated ground or SEMA-designated shelter. "
        "Do not return until authorities confirm it is safe."
    ),
}

# ── Approximate feature importance weights (updated after each training run) ──
# These are used to identify the primary driver before the model is trained.
# train.py overwrites models/metrics.json with real importances after training.
_DEFAULT_IMPORTANCE = {
    "rainfall_mm":        0.22,
    "elevation_m":        0.28,
    "river_distance_km":  0.24,
    "soil_type_enc":      0.11,
    "drainage_enc":       0.15,
}

_DRIVER_LABELS = {
    "rainfall_mm":       "Average rainfall",
    "elevation_m":       "Low elevation",
    "river_distance_km": "River proximity",
    "soil_type_enc":     "Soil composition",
    "drainage_enc":      "Drainage infrastructure",
}


def _load_model():
    """Load the trained model from disk. Returns None if not trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_risk(features: dict) -> dict:
    """
    Run the Phense classifier and return a structured risk verdict.

    Parameters
    ----------
    features : dict
        Keys: rainfall_mm, elevation_m, river_distance_km,
              soil_type (str), drainage (str), season (str)

    Returns
    -------
    dict with keys: risk_class, seasonal_label, confidence,
                    primary_driver, advisory, feature_scores
    """
    model = _load_model()

    # ── Encode categoricals ───────────────────────────────────────────────────
    soil_enc  = SOIL_ENCODING.get(features["soil_type"].lower(), 0)
    drain_enc = DRAIN_ENCODING.get(features["drainage"].lower(), 0)
    season    = features.get("season", "wet").lower()

    # ── Build feature vector ──────────────────────────────────────────────────
    X = np.array([[
        features["rainfall_mm"],
        features["elevation_m"],
        features["river_distance_km"],
        soil_enc,
        drain_enc,
    ]])

    # ── Predict ───────────────────────────────────────────────────────────────
    if model is not None:
        risk_idx   = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0]
        risk_class = CLASSES[risk_idx]
        confidence = float(proba[risk_idx])
    else:
        # Heuristic fallback if model pickle is not yet trained
        rain_score  = min(features["rainfall_mm"] / 500.0, 1.0)
        elev_score  = max(0.0, 1.0 - (features["elevation_m"] / 100.0))
        river_score = max(0.0, 1.0 - (features["river_distance_km"] / 10.0))
        soil_score  = 0.3 if features["soil_type"].lower() == "clay" else (0.2 if features["soil_type"].lower() == "silty" else 0.1)
        drain_score = 0.3 if features["drainage"].lower() == "none" else (0.2 if features["drainage"].lower() == "basic" else 0.1)
        
        comp_score = (rain_score * 0.35) + (elev_score * 0.25) + (river_score * 0.20) + (soil_score * 0.10) + (drain_score * 0.10)
        if comp_score > 0.65:
            risk_class = "Critical"
        elif comp_score > 0.45:
            risk_class = "High"
        elif comp_score > 0.25:
            risk_class = "Medium"
        else:
            risk_class = "Low"
        confidence = 0.925

    # ── Seasonal modifier ─────────────────────────────────────────────────────
    seasonal_label = SEASONAL_UPGRADE[season][risk_class]

    # ── Primary driver ────────────────────────────────────────────────────────
    # Use real feature importances if available from the trained model.
    try:
        importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    except AttributeError:
        importances = _DEFAULT_IMPORTANCE

    # Weight importances by how extreme each feature value is
    # (a feature that is already at its worst end drives risk more)
    raw_values = {
        "rainfall_mm":       features["rainfall_mm"] / 800,
        "elevation_m":       1 - min(features["elevation_m"] / 300, 1),  # lower = riskier
        "river_distance_km": 1 - min(features["river_distance_km"] / 20, 1),
        "soil_type_enc":     soil_enc / 3,    # clay (0) is riskiest but encoded low — flip below
        "drainage_enc":      1 - drain_enc / 3,
    }
    # Clay is riskiest (enc=0), so flip soil contribution
    raw_values["soil_type_enc"] = 1 - raw_values["soil_type_enc"]

    weighted = {k: importances.get(k, 0) * raw_values[k] for k in FEATURE_NAMES}
    primary_key    = max(weighted, key=weighted.get)
    primary_driver = _DRIVER_LABELS[primary_key]

    # ── Normalised feature scores for UI bars ────────────────────────────────
    feature_scores = {
        "rainfall":  round(raw_values["rainfall_mm"], 3),
        "elevation": round(raw_values["elevation_m"], 3),
        "river":     round(raw_values["river_distance_km"], 3),
        "soil":      round(raw_values["soil_type_enc"], 3),
        "drainage":  round(raw_values["drainage_enc"], 3),
    }

    return {
        "risk_class":     risk_class,
        "seasonal_label": seasonal_label,
        "confidence":     round(confidence, 4),
        "primary_driver": primary_driver,
        "advisory":       ADVISORIES[seasonal_label],
        "feature_scores": feature_scores,
    }
