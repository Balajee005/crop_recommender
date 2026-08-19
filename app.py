"""
app.py
------
Production-ready Flask API for the Crop Recommendation System.
Endpoints:
  GET  /           → Serve UI
  POST /predict    → Return crop prediction + confidence scores
  GET  /health     → Health check
  GET  /crops      → List all supported crops
"""

import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, abort

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR),
            static_folder=str(STATIC_DIR))

# ── Load artefacts at startup ─────────────────────────────────────────────────
try:
    MODEL   = joblib.load(MODEL_DIR / "ensemble_model.joblib")
    SCALER  = joblib.load(MODEL_DIR / "scaler.joblib")
    ENCODER = joblib.load(MODEL_DIR / "label_encoder.joblib")
    CLASSES = list(ENCODER.classes_)
    print(f"[App] Models loaded. Supports {len(CLASSES)} crops.")
except Exception as e:
    print(f"[App] FATAL — Could not load model artefacts: {e}")
    sys.exit(1)

# ── Feature schema ────────────────────────────────────────────────────────────
FEATURE_SCHEMA = {
    "N":           {"min": 0,   "max": 200,  "label": "Nitrogen (N)"},
    "P":           {"min": 0,   "max": 200,  "label": "Phosphorus (P)"},
    "K":           {"min": 0,   "max": 250,  "label": "Potassium (K)"},
    "temperature": {"min": 0.0, "max": 60.0, "label": "Temperature (°C)"},
    "humidity":    {"min": 0.0, "max": 100.0,"label": "Humidity (%)"},
    "ph":          {"min": 0.0, "max": 14.0, "label": "pH"},
    "rainfall":    {"min": 0.0, "max": 500.0,"label": "Rainfall (mm)"},
}
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Crop emoji map ────────────────────────────────────────────────────────────
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘",
    "blackgram": "🫘", "lentil": "🫘", "pomegranate": "🍎",
    "banana": "🍌", "mango": "🥭", "grapes": "🍇",
    "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍑", "coconut": "🥥",
    "cotton": "🌿", "jute": "🌱", "coffee": "☕",
}


def validate_input(data: dict) -> tuple[dict, list]:
    """Parse and validate incoming JSON. Returns (parsed_values, errors)."""
    values = {}
    errors = []

    for feature, meta in FEATURE_SCHEMA.items():
        if feature not in data:
            errors.append(f"Missing field: '{feature}'")
            continue
        try:
            val = float(data[feature])
        except (ValueError, TypeError):
            errors.append(f"'{feature}' must be a number.")
            continue
        if val < meta["min"] or val > meta["max"]:
            errors.append(
                f"'{feature}' must be between {meta['min']} and {meta['max']} "
                f"(got {val})."
            )
            continue
        values[feature] = val

    return values, errors


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", crops=CLASSES)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "crops_supported": len(CLASSES)})


@app.route("/crops")
def crops():
    return jsonify({"crops": CLASSES, "count": len(CLASSES)})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    values, errors = validate_input(body)
    if errors:
        return jsonify({"errors": errors}), 422

    # Build feature vector in correct order
    X = np.array([[values[f] for f in FEATURE_ORDER]], dtype=float)

    # Scale
    X_scaled = SCALER.transform(X)

    # Predict
    pred_idx   = MODEL.predict(X_scaled)[0]
    probas     = MODEL.predict_proba(X_scaled)[0]
    crop_name  = ENCODER.inverse_transform([pred_idx])[0]
    confidence = float(probas[pred_idx])

    # Top-3 crops with probabilities
    top3_idx   = np.argsort(probas)[::-1][:3]
    top3       = [
        {
            "crop":       ENCODER.inverse_transform([i])[0],
            "probability": round(float(probas[i]), 4),
            "emoji":      CROP_EMOJI.get(ENCODER.inverse_transform([i])[0], "🌿"),
        }
        for i in top3_idx
    ]

    return jsonify({
        "predicted_crop": crop_name,
        "confidence":     round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "emoji":          CROP_EMOJI.get(crop_name, "🌿"),
        "top_3":          top3,
        "input_received": values,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[App] Starting Crop Recommendation API on port {port} …")
    print(f"[App] Open  http://localhost:{port}  in your browser.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
