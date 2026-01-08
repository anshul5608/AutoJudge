"""
utils.py
----------------------------------
Shared utilities for AutoJudge.

Responsibilities:
- Centralized path management
- Safe loading of trained models
- Prediction helper functions
- Label decoding
"""

from pathlib import Path
from typing import Tuple
import joblib
import re

# ============================================================
# Project paths (single source of truth)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"

CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "regressor.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

# ============================================================
# Label mappings
# ============================================================

CLASS_ID_TO_NAME = {
    0: "Easy",
    1: "Medium",
    2: "Hard",
}

# ============================================================
# Model Loaders
# ============================================================

def load_classifier():
    """
    Load trained classification model.
    """

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found at {CLASSIFIER_PATH}"
        )

    return joblib.load(CLASSIFIER_PATH)


def load_regressor():
    """
    Load trained regression model.
    """

    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Regressor model not found at {REGRESSOR_PATH}"
        )

    return joblib.load(REGRESSOR_PATH)


def load_vectorizer():
    """
    Load TF-IDF vectorizer.
    """

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {VECTORIZER_PATH}"
        )

    return joblib.load(VECTORIZER_PATH)


# ============================================================
# Input Sanitization
# ============================================================

def sanitize_input(text: str) -> str:
    """
    Sanitize user-provided text input.

    Used before prediction.
    """

    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# Prediction Helpers
# ============================================================

def decode_class_label(label_id: int) -> str:
    """
    Convert numeric class to human-readable label.
    """

    return CLASS_ID_TO_NAME.get(label_id, "Unknown")


def predict_difficulty(
    text_vector
) -> Tuple[str, float]:
    """
    Predict difficulty class and score for a problem.

    Args:
        text_vector: Final feature vector (sparse matrix)

    Returns:
        (difficulty_class, difficulty_score)
    """

    classifier = load_classifier()
    regressor = load_regressor()

    class_id = classifier.predict(text_vector)[0]
    score = regressor.predict(text_vector)[0]

    difficulty_name = decode_class_label(class_id)

    return difficulty_name, float(score)


# ============================================================
# Debug / Sanity Test
# ============================================================

if __name__ == "__main__":
    print("Utils module sanity check")
    print("Project root:", PROJECT_ROOT)
    print("Models directory:", MODEL_DIR)
