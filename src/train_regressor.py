"""
train_regressor.py
----------------------------------
Train difficulty SCORE regressor
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_regression_features
from preprocess import prepare_regression_targets

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

REGRESSOR_PATH = MODEL_DIR / "regressor.pkl"

# ============================================================
# Training
# ============================================================

def train_regressor():
    print("Building regression features...")
    X = build_regression_features()

    print("Loading targets...")
    y = prepare_regression_targets()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("Training regressor...")
    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = reg.predict(X_test)

    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    joblib.dump(reg, REGRESSOR_PATH)
    print(f"Regressor saved at: {REGRESSOR_PATH}")

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_regressor()
