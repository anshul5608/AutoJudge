"""
app.py
----------------------------------
Flask web application for AutoJudge.

Responsibilities:
- Serve UI
- Accept user input
- Run ML inference
- Display predictions with confidence awareness

This file is inference-only.
NO training logic here.
"""

from flask import Flask, render_template, request
from pathlib import Path
import numpy as np
import sys

from scipy.sparse import hstack

# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# ============================================================
# Imports from project
# ============================================================

from utils import (
    load_classifier,
    load_regressor,
    load_vectorizer,
    decode_class_label,
    sanitize_input,
)

from features import extract_numeric_features

# ============================================================
# Flask App Configuration
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# ============================================================
# Load ML models ONCE
# ============================================================

print("Loading ML models...")

classifier = load_classifier()
regressor = load_regressor()
vectorizer = load_vectorizer()

print("Models loaded successfully.")

# ============================================================
# Helper: bucket score → difficulty (stable interpretation)
# ============================================================

def difficulty_from_score(score: float) -> str:
    if score < 3.5:
        return "Easy"
    elif score < 6.0:
        return "Medium"
    else:
        return "Hard"

# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    warning = None

    if request.method == "POST":
        # ----------------------------------------------------
        # 1. Read & sanitize input
        # ----------------------------------------------------
        raw_text = request.form.get("problem", "")
        clean_text = sanitize_input(raw_text)

        if not clean_text:
            warning = "Please paste a valid problem statement."
        else:
            # ------------------------------------------------
            # 2. Input quality warning (UX improvement)
            # ------------------------------------------------
            if len(clean_text.split()) < 50:
                warning = (
                    "Prediction may be inaccurate. "
                    "Please paste the full problem statement."
                )

            # ------------------------------------------------
            # 3. CLASSIFIER inference (TEXT ONLY)
            # ------------------------------------------------
            X_clf = vectorizer.transform([clean_text])

            class_id = classifier.predict(X_clf)[0]
            primary_class = decode_class_label(class_id)

            # --- confidence + top-2 handling (if supported) ---
            secondary_class = None
            confidence_note = None

            if hasattr(classifier, "decision_function"):
                scores = classifier.decision_function(X_clf)[0]

                # Top-2 classes
                top2 = np.argsort(scores)[-2:][::-1]
                primary_class = decode_class_label(top2[0])
                secondary_class = decode_class_label(top2[1])

                # Confidence heuristic
                confidence_gap = scores[top2[0]] - scores[top2[1]]
                if confidence_gap < 0.5:
                    confidence_note = "Low confidence"

            # ------------------------------------------------
            # 4. REGRESSOR inference (TEXT + NUMERIC)
            # ------------------------------------------------
            X_text = vectorizer.transform([clean_text])
            X_num = extract_numeric_features([clean_text])
            X_reg = hstack([X_text, X_num])

            score = float(regressor.predict(X_reg)[0])

            # ------------------------------------------------
            # 5. Score-driven final difficulty (more stable)
            # ------------------------------------------------
            score_based_class = difficulty_from_score(score)

            prediction = {
                "class": score_based_class,
                "primary_ml_class": primary_class,
                "secondary_ml_class": secondary_class,
                "score": round(score, 2),
                "confidence_note": confidence_note,
            }

    return render_template(
        "index.html",
        prediction=prediction,
        warning=warning,
    )

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
