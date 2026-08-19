"""
preprocessing.py
----------------
Feature engineering, scaling, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import joblib
from pathlib import Path


FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"


def encode_labels(df: pd.DataFrame):
    """Encode string labels to integers, return (df_encoded, encoder)."""
    le = LabelEncoder()
    df = df.copy()
    df["label_enc"] = le.fit_transform(df[TARGET_COL])
    print(f"[Preprocessing] Classes: {list(le.classes_)}")
    return df, le


def split_data(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42):
    """Stratified 70/30 train-test split."""
    X = df[FEATURE_COLS].values
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray, save_path: str = None):
    """Fit StandardScaler on train, transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if save_path:
        joblib.dump(scaler, save_path)
        print(f"[Preprocessing] Scaler saved to {save_path}")
    return X_train_scaled, X_test_scaled, scaler


def check_imbalance(y_train: np.ndarray, threshold: float = 2.0) -> bool:
    """Return True if class imbalance ratio exceeds threshold."""
    unique, counts = np.unique(y_train, return_counts=True)
    ratio = counts.max() / counts.min()
    print(f"[Preprocessing] Class imbalance ratio: {ratio:.2f} "
          f"({'IMBALANCED' if ratio > threshold else 'BALANCED'})")
    return ratio > threshold
