"""
Train and Save the Verified 1,457-Feature LogisticRegression Model

Recreates the exact training pipeline from the benchmark that achieved 49.14% 
FER2013 PublicTest accuracy.

Configuration from benchmark.py:
- C=1.0
- max_iter=2000
- random_state=42
- class_weight="balanced"
- solver="lbfgs"
- n_jobs=-1

Features: 1,457 expanded MediaPipe features (after cleaning)
Dataset: training/fer2013_features_expanded/
"""

import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Label mapping
TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]


def clean_features(X_train, X_val, X_test, feature_names):
    """Remove constant and NaN-only features - exact copy from benchmark.py"""
    X_all = np.vstack([X_train, X_val, X_test])
    
    keep_indices = []
    keep_names = []
    
    for i in range(X_all.shape[1]):
        col = X_all[:, i]
        nan_count = np.sum(np.isnan(col))
        std = np.std(col[~np.isnan(col)]) if nan_count < len(col) else 0
        
        if nan_count < len(col) and std > 1e-10:
            keep_indices.append(i)
            if i < len(feature_names):
                keep_names.append(feature_names[i])
            else:
                keep_names.append(f"feature_{i}")
        else:
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            print(f"  Dropping feature {i}: {name} (NaN={nan_count}, std={std:.6f})")
    
    return X_train[:, keep_indices], X_val[:, keep_indices], X_test[:, keep_indices], keep_names


def evaluate_model(model, X, y, target_names=TARGET_NAMES) -> Dict[str, Any]:
    """Comprehensive model evaluation - exact copy from benchmark.py"""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision_w = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y, y_pred, average="weighted", zero_division=0)
    
    precision_macro = precision_score(y, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    
    precision_per = precision_score(y, y_pred, average=None, zero_division=0)
    recall_per = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y, y_pred)
    
    per_class = {}
    for i, name in enumerate(target_names):
        per_class[name] = {
            "precision": float(precision_per[i]) if i < len(precision_per) else 0.0,
            "recall": float(recall_per[i]) if i < len(recall_per) else 0.0,
            "f1": float(f1_per[i]) if i < len(f1_per) else 0.0,
            "support": int(np.sum(y == i))
        }
    
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


def main():
    print("="*60)
    print("TRAINING VERIFIED 1,457-FEATURE LOGISTIC REGRESSION")
    print("="*60)
    
    # Load expanded data
    print("\nLoading expanded 1457-feature dataset...")
    X_train = np.load('training/fer2013_features_expanded/X_train.npy')
    X_val = np.load('training/fer2013_features_expanded/X_val.npy')
    X_test = np.load('training/fer2013_features_expanded/X_test.npy')
    y_train = np.load('training/fer2013_features_expanded/y_train.npy')
    y_val = np.load('training/fer2013_features_expanded/y_val.npy')
    y_test = np.load('training/fer2013_features_expanded/y_test.npy')
    
    with open('training/fer2013_features_expanded/metadata.json', 'r') as f:
        meta = json.load(f)
    
    feature_names = meta['feature_names']
    
    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Clean features (remove constant/NaN)
    print("\nCleaning features (removing constant/NaN)...")
    X_train, X_val, X_test, feature_names = clean_features(X_train, X_val, X_test, feature_names)
    print(f"Cleaned feature count: {X_train.shape[1]}")
    
    # Re-scale after cleaning (exact benchmark pipeline)
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train LogisticRegression (exact benchmark configuration)
    print("\nTraining LogisticRegression...")
    print("  C=1.0, max_iter=2000, random_state=42")
    print("  class_weight='balanced', solver='lbfgs', n_jobs=-1")
    
    start = time.time()
    lr = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate on validation
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(lr, X_val, y_val)
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Val Macro F1: {val_metrics['f1_macro']:.4f}")
    print(f"Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
    
    # Final evaluation on test set
    print("\nEvaluating on held-out test set (PublicTest)...")
    test_metrics = evaluate_model(lr, X_test, y_test)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Test Macro Precision:   {test_metrics['precision_macro']:.4f}")
    print(f"Test Macro Recall:      {test_metrics['recall_macro']:.4f}")
    print(f"Test Macro F1:          {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted Precision:{test_metrics['precision_weighted']:.4f}")
    print(f"Test Weighted Recall:   {test_metrics['recall_weighted']:.4f}")
    print(f"Test Weighted F1:       {test_metrics['f1_weighted']:.4f}")
    
    print("\nPer-class Test Results:")
    for cls in TARGET_NAMES:
        m = test_metrics['per_class'][cls]
        print(f"  {cls:10s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['support']}")
    
    print("\nConfusion Matrix:")
    cm = np.array(test_metrics['confusion_matrix'])
    header = "      " + "  ".join(f"{n[:4]:>4}" for n in TARGET_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{TARGET_NAMES[i][:4]:>4} " + "  ".join(f"{v:>4}" for v in row))
    
    # Verify against benchmark results
    benchmark_accuracy = 0.49139414802065406
    benchmark_macro_f1 = 0.4860619690380382
    benchmark_weighted_f1 = 0.49120014653816585
    
    print(f"\n{'='*60}")
    print("VERIFICATION AGAINST BENCHMARK")
    print(f"{'='*60}")
    print(f"Benchmark Accuracy:     {benchmark_accuracy:.4f}")
    print(f"Our Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"Difference:             {abs(test_metrics['accuracy'] - benchmark_accuracy):.6f}")
    print(f"Benchmark Macro F1:     {benchmark_macro_f1:.4f}")
    print(f"Our Macro F1:           {test_metrics['f1_macro']:.4f}")
    print(f"Difference:             {abs(test_metrics['f1_macro'] - benchmark_macro_f1):.6f}")
    print(f"Benchmark Weighted F1:  {benchmark_weighted_f1:.4f}")
    print(f"Our Weighted F1:        {test_metrics['f1_weighted']:.4f}")
    print(f"Difference:             {abs(test_metrics['f1_weighted'] - benchmark_weighted_f1):.6f}")
    
    # Check if matches closely (within 0.01%)
    tol = 1e-4
    matches = (
        abs(test_metrics['accuracy'] - benchmark_accuracy) < tol and
        abs(test_metrics['f1_macro'] - benchmark_macro_f1) < tol and
        abs(test_metrics['f1_weighted'] - benchmark_weighted_f1) < tol
    )
    
    if matches:
        print("\nVERIFICATION PASSED: Results match benchmark within tolerance")
    else:
        print("\nVERIFICATION WARNING: Results differ from benchmark")
        print("    This may be due to random state or sklearn version differences")
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    models_dir = Path('training/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'logistic_regression_expanded_1457.joblib'
    scaler_path = models_dir / 'scaler_expanded_1457.joblib'
    metadata_path = models_dir / 'logistic_regression_expanded_1457_metadata.json'
    
    joblib.dump(lr, model_path)
    print(f"Model saved: {model_path}")
    
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # Create metadata
    metadata = {
        "model_type": "LogisticRegression",
        "feature_dim": X_train.shape[1],
        "feature_names": feature_names,
        "dataset_path": "training/fer2013_features_expanded/",
        "dataset_version": "expanded_1457_v1",
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "preprocessing": {
            "clean_features": True,
            "scaler": "StandardScaler",
            "scaler_fitted_on": "training_data_only"
        },
        "hyperparameters": {
            "C": 1.0,
            "max_iter": 2000,
            "random_state": 42,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "n_jobs": -1
        },
        "random_seed": 42,
        "test_accuracy": float(test_metrics['accuracy']),
        "macro_f1": float(test_metrics['f1_macro']),
        "weighted_f1": float(test_metrics['f1_weighted']),
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_reference_accuracy": benchmark_accuracy,
        "benchmark_reference_macro_f1": benchmark_macro_f1,
        "benchmark_reference_weighted_f1": benchmark_weighted_f1,
        "verification_passed": matches
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # Quick reload verification
    print("\n" + "="*60)
    print("RELOAD VERIFICATION")
    print(f"{'='*60}")
    
    # Reload and verify
    lr_reload = joblib.load(model_path)
    scaler_reload = joblib.load(scaler_path)
    with open(metadata_path, 'r') as f:
        meta_reload = json.load(f)
    
    # Test prediction
    test_sample = X_test[:1]
    pred_orig = lr.predict(test_sample)[0]
    pred_reload = lr_reload.predict(test_sample)[0]
    
    proba_orig = lr.predict_proba(test_sample)[0]
    proba_reload = lr_reload.predict_proba(test_sample)[0]
    
    print(f"Model loads: OK")
    print(f"Scaler loads: OK")
    print(f"Metadata loads: OK")
    print(f"Feature dimension: {meta_reload['feature_dim']}")
    print(f"Prediction matches: {pred_orig == pred_reload}")
    print(f"Probabilities match: {np.allclose(proba_orig, proba_reload)}")
    
    if pred_orig == pred_reload and np.allclose(proba_orig, proba_reload):
        print("\nRELOAD VERIFICATION PASSED")
    else:
        print("\nRELOAD VERIFICATION FAILED")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model:     {model_path}")
    print(f"Scaler:    {scaler_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"Features:  {X_train.shape[1]}")
    print(f"Test Acc:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1:  {test_metrics['f1_macro']*100:.2f}%")
    print(f"Wghtd F1:  {test_metrics['f1_weighted']*100:.2f}%")
    
    return matches


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)