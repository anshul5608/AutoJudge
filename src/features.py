"""
features.py
----------------------------------
Feature engineering for AutoJudge.

Rules:
- TF-IDF vectorizer is fitted ONCE
- Classification uses ONLY text features
- Regression uses text + numeric features
"""

from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import prepare_text_corpus

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

# ============================================================
# Numeric features (used ONLY for regression)
# ============================================================

def extract_numeric_features(texts):
    features = []
    for text in texts:
        features.append([
            len(text),                 # text length
            len(text.split()),         # word count
        ])
    return np.array(features, dtype=np.float32)

# ============================================================
# Vectorizer setup
# ============================================================

def fit_vectorizer():
    texts = prepare_text_corpus()

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return X


def load_vectorizer():
    if not VECTORIZER_PATH.exists():
        raise RuntimeError(
            "Vectorizer not found. Run: python src/features.py"
        )
    return joblib.load(VECTORIZER_PATH)

# ============================================================
# Public feature builders
# ============================================================

def build_text_features():
    texts = prepare_text_corpus()
    vectorizer = load_vectorizer()
    return vectorizer.transform(texts)


def build_regression_features():
    texts = prepare_text_corpus()
    vectorizer = load_vectorizer()

    X_text = vectorizer.transform(texts)
    X_num = extract_numeric_features(texts)

    return hstack([X_text, csr_matrix(X_num)])

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("Fitting TF-IDF vectorizer...")
    X = fit_vectorizer()
    print("Vectorizer fitted.")
    print("Feature shape:", X.shape)
