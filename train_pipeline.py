"""
train_pipeline.py
-----------------
Entry point for the full ML training pipeline.
Run: python train_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader    import load_data, summarize
from src.preprocessing  import encode_labels, split_data, scale_features, check_imbalance
from src.model_trainer  import build_and_train

import joblib

DATASET_PATH = "crop_recommendation.csv"
MODEL_DIR    = "models"


def run_pipeline():
    print("=" * 70)
    print("  Ensemble-Based Crop Recommendation System — Training Pipeline")
    print("=" * 70)

    # ── Step 1: Load & EDA ────────────────────────────────────────────────
    print("\n[Step 1] Loading data …")
    df = load_data(DATASET_PATH)
    summarize(df)

    # ── Step 2: Encode + Split ────────────────────────────────────────────
    print("\n[Step 2] Encoding labels and splitting data …")
    df_enc, le = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df_enc)

    # ── Step 3: Scale ─────────────────────────────────────────────────────
    print("\n[Step 3] Scaling features …")
    scaler_path = f"{MODEL_DIR}/scaler.joblib"
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test,
                                                   save_path=scaler_path)

    # ── Step 4: Check imbalance ───────────────────────────────────────────
    print("\n[Step 4] Checking class balance …")
    imbalanced = check_imbalance(y_train)
    if imbalanced:
        print("[Warning] Imbalance detected — using class_weight='balanced' in models.")
    else:
        print("[OK] Dataset is balanced — SMOTE not required.")

    # ── Step 5: Train, tune, evaluate ─────────────────────────────────────
    print("\n[Step 5] Training and evaluating models …")
    results = build_and_train(
        X_train_s, y_train, X_test_s, y_test,
        label_encoder=le,
        model_dir=MODEL_DIR,
        n_iter=15,
        cv_folds=10,
    )

    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    ensemble_acc = results["eval_results"]["SoftVotingEnsemble"]["accuracy"]
    print(f"\n  Final Ensemble Test Accuracy: {ensemble_acc:.4f} "
          f"({ensemble_acc*100:.2f}%)")
    print(f"  Model saved to: {results['paths']['ensemble']}")
    print(f"  Scaler saved to: {scaler_path}")
    print(f"  Encoder saved to: {results['paths']['encoder']}")
    print("\n  Run `python app.py` to start the Flask API.\n")

    return results


if __name__ == "__main__":
    run_pipeline()
