"""
data_loader.py
--------------
Handles loading and initial validation of the crop recommendation dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the crop recommendation CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(path)

    expected_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"[DataLoader] Loaded {len(df)} rows, {df['label'].nunique()} unique crops.")
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print EDA summary."""
    print("\n=== Dataset Summary ===")
    print(df.describe().round(3))
    print("\n=== Class Distribution ===")
    counts = df["label"].value_counts()
    for crop, count in counts.items():
        print(f"  {crop:<20} {count}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nImbalance ratio (max/min): "
          f"{counts.max() / counts.min():.2f}")
