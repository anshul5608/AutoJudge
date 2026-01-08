"""
preprocess.py
----------------------------------
Central preprocessing module for AutoJudge.

Responsibilities:
- Load dataset from disk
- Clean and normalize text
- Combine text fields into a single document
- Encode difficulty labels

This module should NOT do:
- Feature engineering
- Model training
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# ============================================================
# Project paths (absolute & safe)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "problems_data.jsonl"

# ============================================================
# Constants
# ============================================================

TEXT_FIELDS = [
    "title",
    "description",
    "input_description",
    "output_description",
]

LABEL_MAPPING = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2,
}

# ============================================================
# Text Cleaning Utilities
# ============================================================

def clean_text(text: str) -> str:
    """
    Perform standard NLP cleaning on input text.

    Steps:
    - Lowercasing
    - Remove HTML tags
    - Normalize whitespace
    - Remove non-alphanumeric symbols (keeps math/logical chars)
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove unwanted symbols (keep numbers and common operators)
    text = re.sub(r"[^a-z0-9\s\+\-\*/=<>()]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def merge_text_fields(sample: Dict) -> str:
    """
    Merge all relevant text fields of a problem into one string.
    """

    parts = []

    for field in TEXT_FIELDS:
        value = sample.get(field, "")
        parts.append(clean_text(value))

    return " ".join(parts)


# ============================================================
# Data Loading
# ============================================================

def load_raw_data(path: Path = DATA_FILE) -> List[Dict]:
    """
    Load JSONL dataset safely.

    Returns:
        List of dictionaries (one per problem)
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num}"
                ) from e

    if not data:
        raise ValueError("Dataset loaded but is empty")

    return data


# ============================================================
# Public Preprocessing APIs
# ============================================================

def prepare_text_corpus() -> List[str]:
    """
    Returns:
        List of cleaned, merged problem descriptions.
    """

    raw_data = load_raw_data()
    corpus = [merge_text_fields(sample) for sample in raw_data]
    return corpus


def prepare_classification_labels() -> List[int]:
    """
    Returns:
        Encoded difficulty class labels (0, 1, 2)
    """

    raw_data = load_raw_data()
    labels = []

    for sample in raw_data:
        label_raw = sample.get("problem_class")
        
        if not isinstance(label_raw, str):
            raise ValueError(f"Invalid label type: {label_raw}")

        label = label_raw.strip().lower()

        LABEL_NORMALIZED = {
            "easy": 0,
            "medium": 1,
            "hard": 2,
        }

        if label not in LABEL_NORMALIZED:
            raise ValueError(f"Unknown label encountered: {label_raw}")

        labels.append(LABEL_NORMALIZED[label])


    return labels


def prepare_regression_targets() -> List[float]:
    """
    Returns:
        Difficulty scores for regression.
    """

    raw_data = load_raw_data()
    scores = []

    for sample in raw_data:
        score = sample.get("problem_score")

        if score is None:
            raise ValueError("Missing problem_score in dataset")

        scores.append(float(score))

    return scores


# ============================================================
# Debug / Standalone Test
# ============================================================

if __name__ == "__main__":
    print("Running preprocess sanity check...\n")

    texts = prepare_text_corpus()
    y_class = prepare_classification_labels()
    y_score = prepare_regression_targets()

    print(f"Total samples        : {len(texts)}")
    print(f"Sample text (short)  : {texts[0][:200]}...")
    print(f"Sample class label   : {y_class[0]}")
    print(f"Sample score         : {y_score[0]}")
