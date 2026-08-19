"""
model_trainer.py
----------------
Defines, tunes, cross-validates, and builds the soft-voting ensemble.
Uses RandomForestClassifier, ExtraTreesClassifier, and
GradientBoostingClassifier (boosting equivalent of XGBoost).
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ─── Hyperparameter search spaces ─────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

ET_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

GB_PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 5],
}


def tune_model(estimator, param_grid: dict, X_train, y_train,
               n_iter: int = 15, cv: int = 5, n_jobs: int = -1,
               random_state: int = 42, label: str = "Model"):
    """RandomizedSearchCV tuning helper."""
    print(f"\n[Tuning] {label} — {n_iter} iterations × {cv}-fold CV …")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="accuracy",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"[Tuning] {label} best CV accuracy: {search.best_score_:.4f}")
    print(f"[Tuning] {label} best params: {search.best_params_}")
    return search.best_estimator_


def cross_validate_model(model, X, y, cv: int = 10, label: str = "Model"):
    """10-fold stratified cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"[CV] {label:30s} | Mean: {scores.mean():.4f} | Std: {scores.std():.4f} "
          f"| Min: {scores.min():.4f} | Max: {scores.max():.4f}")
    return scores


def evaluate_model(model, X_test, y_test, label_names=None):
    """Full metric evaluation."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return metrics, y_pred


def build_and_train(X_train, y_train, X_test, y_test,
                    label_encoder, model_dir: str = "models",
                    n_iter: int = 15, cv_folds: int = 10):
    """
    Full pipeline:
      1. Tune each base learner
      2. Cross-validate individually
      3. Build soft-voting ensemble
      4. Evaluate all models
      5. Save ensemble + label encoder
    Returns dict of results.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Tune base learners ──────────────────────────────────────────────
    rf_best = tune_model(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        RF_PARAM_GRID, X_train, y_train,
        n_iter=n_iter, label="RandomForest"
    )
    et_best = tune_model(
        ExtraTreesClassifier(random_state=42, class_weight="balanced"),
        ET_PARAM_GRID, X_train, y_train,
        n_iter=n_iter, label="ExtraTrees"
    )
    gb_best = tune_model(
        GradientBoostingClassifier(random_state=42),
        GB_PARAM_GRID, X_train, y_train,
        n_iter=n_iter, label="GradientBoosting"
    )

    # ── 2. Soft-Voting Ensemble ────────────────────────────────────────────
    print("\n[Ensemble] Building Soft-Voting Ensemble …")
    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_best),
            ("et", et_best),
            ("gb", gb_best),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    print("[Ensemble] Training complete.")

    # ── 3. Cross-validate all ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Model':<30} | {'Mean':>6} | {'Std':>6} | {'Min':>6} | {'Max':>6}")
    print(f"{'─'*70}")

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    cv_results = {}
    for name, model in [("RandomForest", rf_best), ("ExtraTrees", et_best),
                        ("GradientBoosting", gb_best), ("SoftVotingEnsemble", ensemble)]:
        scores = cross_validate_model(model, X_full, y_full, cv=cv_folds, label=name)
        cv_results[name] = scores

    # ── 4. Evaluate on test set ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Model':<30} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print(f"{'─'*70}")

    eval_results = {}
    for name, model in [("RandomForest", rf_best), ("ExtraTrees", et_best),
                        ("GradientBoosting", gb_best), ("SoftVotingEnsemble", ensemble)]:
        m, _ = evaluate_model(model, X_test, y_test, label_names=label_encoder.classes_)
        eval_results[name] = m
        print(f"{name:<30} | {m['accuracy']:>7.4f} | {m['precision']:>7.4f} | "
              f"{m['recall']:>7.4f} | {m['f1']:>7.4f}")

    # ── 5. Detailed ensemble report ────────────────────────────────────────
    y_pred_ensemble = ensemble.predict(X_test)
    print(f"\n{'─'*70}")
    print("Ensemble Classification Report:")
    print(classification_report(y_test, y_pred_ensemble,
                                target_names=label_encoder.classes_))

    # ── 6. Save artifacts ─────────────────────────────────────────────────
    ensemble_path = Path(model_dir) / "ensemble_model.joblib"
    le_path       = Path(model_dir) / "label_encoder.joblib"
    joblib.dump(ensemble,       ensemble_path)
    joblib.dump(label_encoder,  le_path)
    print(f"\n[Save] Ensemble  → {ensemble_path}")
    print(f"[Save] Encoder   → {le_path}")

    return {
        "models": {
            "rf": rf_best, "et": et_best,
            "gb": gb_best, "ensemble": ensemble,
        },
        "cv_results":   cv_results,
        "eval_results": eval_results,
        "paths": {
            "ensemble": str(ensemble_path),
            "encoder":  str(le_path),
        },
    }
