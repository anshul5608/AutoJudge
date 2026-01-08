"""
train_classifier.py
----------------------------------
Train difficulty CLASSIFIER (Easy / Medium / Hard)
"""

from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report

from features import build_text_features
from preprocess import prepare_classification_labels

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"

# ============================================================
# Training
# ============================================================

def train_classifier():
    print("Building text features...")
    X = build_text_features()

    print("Loading labels...")
    y = prepare_classification_labels()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training classifier...")
    clf = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, CLASSIFIER_PATH)
    print(f"Classifier saved at: {CLASSIFIER_PATH}")

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_classifier()
