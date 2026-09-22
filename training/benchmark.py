"""
Facial Expression Recognition Benchmark Script

Benchmarks multiple classifiers on:
1. Baseline 48-engineered features (current production)
2. Expanded MediaPipe features (1457 features: landmarks + blendshapes + head pose)

Evaluates on held-out PublicTest set only after validation-based model selection.
"""

import sys
import time
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping")

# Label mapping
TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
FER2013_LABEL_MAP = {i: name for i, name in enumerate(TARGET_NAMES)}


def clean_features(X_train, X_val, X_test, feature_names):
    """Remove constant and NaN-only features."""
    # Combine to check for constant/NaN features
    X_all = np.vstack([X_train, X_val, X_test])
    
    keep_indices = []
    keep_names = []
    
    for i in range(X_all.shape[1]):
        col = X_all[:, i]
        nan_count = np.sum(np.isnan(col))
        std = np.std(col[~np.isnan(col)]) if nan_count < len(col) else 0
        
        # Keep if not all NaN and not constant
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


def load_baseline_data():
    """Load baseline 48-feature dataset."""
    X_train = np.load('training/fer2013_features_v2/X_train.npy')
    X_val = np.load('training/fer2013_features_v2/X_val.npy')
    X_test = np.load('training/fer2013_features_v2/X_test.npy')
    y_train = np.load('training/fer2013_features_v2/y_train.npy')
    y_val = np.load('training/fer2013_features_v2/y_val.npy')
    y_test = np.load('training/fer2013_features_v2/y_test.npy')
    scaler = joblib.load('training/fer2013_features_v2/scaler.joblib')
    
    with open('training/fer2013_features_v2/metadata.json', 'r') as f:
        meta = json.load(f)
    
    feature_names = meta['feature_names'][:48]
    
    # Clean features (remove constant/NaN)
    X_train, X_val, X_test, feature_names = clean_features(X_train, X_val, X_test, feature_names)
    
    # Re-scale after cleaning
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def load_expanded_data():
    """Load expanded 1457-feature dataset."""
    X_train = np.load('training/fer2013_features_expanded/X_train.npy')
    X_val = np.load('training/fer2013_features_expanded/X_val.npy')
    X_test = np.load('training/fer2013_features_expanded/X_test.npy')
    y_train = np.load('training/fer2013_features_expanded/y_train.npy')
    y_val = np.load('training/fer2013_features_expanded/y_val.npy')
    y_test = np.load('training/fer2013_features_expanded/y_test.npy')
    scaler = joblib.load('training/fer2013_features_expanded/scaler.joblib')
    
    with open('training/fer2013_features_expanded/metadata.json', 'r') as f:
        meta = json.load(f)
    
    feature_names = meta['feature_names']
    
    # Clean features (remove constant/NaN)
    X_train, X_val, X_test, feature_names = clean_features(X_train, X_val, X_test, feature_names)
    
    # Re-scale after cleaning
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def evaluate_model(model, X, y, target_names=TARGET_NAMES) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
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


def train_random_forest(X_train, y_train, **kwargs):
    """Train Random Forest classifier."""
    defaults = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
        "verbose": 0
    }
    defaults.update(kwargs)
    rf = RandomForestClassifier(**defaults)
    rf.fit(X_train, y_train)
    return rf


def train_svm(X_train, y_train, **kwargs):
    """Train SVM with calibration for probabilities."""
    defaults = {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "random_state": 42,
        "class_weight": "balanced",
        "probability": False
    }
    defaults.update(kwargs)
    svm = SVC(**defaults)
    calibrated = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def train_logistic_regression(X_train, y_train, **kwargs):
    """Train Logistic Regression."""
    defaults = {
        "C": 1.0,
        "max_iter": 2000,
        "random_state": 42,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "n_jobs": -1
    }
    defaults.update(kwargs)
    lr = LogisticRegression(**defaults)
    lr.fit(X_train, y_train)
    return lr


def train_mlp(X_train, y_train, **kwargs):
    """Train MLP classifier."""
    defaults = {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 64,
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "verbose": False
    }
    defaults.update(kwargs)
    mlp = MLPClassifier(**defaults)
    mlp.fit(X_train, y_train)
    return mlp


def train_xgboost(X_train, y_train, **kwargs):
    """Train XGBoost classifier."""
    if not HAS_XGBOOST:
        return None
    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "verbosity": 0
    }
    defaults.update(kwargs)
    xgb = XGBClassifier(**defaults)
    xgb.fit(X_train, y_train)
    return xgb


def benchmark_dataset(name: str, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Run benchmark on a dataset with multiple classifiers."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {name}")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    results = {}
    
    # Define models to test
    models = {
        "RandomForest": (train_random_forest, {}),
        "SVM": (train_svm, {}),
        "LogisticRegression": (train_logistic_regression, {}),
        "MLP": (train_mlp, {}),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = (train_xgboost, {})
    
    best_model = None
    best_val_f1 = -1
    best_model_name = ""
    
    for model_name, (train_fn, kwargs) in models.items():
        print(f"\n--- Training {model_name} ---")
        start = time.time()
        try:
            model = train_fn(X_train, y_train, **kwargs)
            train_time = time.time() - start
            
            if model is None:
                print(f"  {model_name}: Not available")
                continue
            
            # Evaluate on validation
            val_metrics = evaluate_model(model, X_val, y_val)
            
            print(f"  Train time: {train_time:.2f}s")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Macro F1: {val_metrics['f1_macro']:.4f}")
            print(f"  Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
            
            # Track best model on validation
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                best_model = model
                best_model_name = model_name
            
            results[model_name] = {
                "train_time": train_time,
                "val_metrics": val_metrics,
                "model": model
            }
            
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}
    
    print(f"\n>>> Best model on validation: {best_model_name} (Macro F1: {best_val_f1:.4f})")
    
    # Final evaluation on test set using best model
    if best_model is not None:
        print(f"\n--- Final Test Evaluation ({best_model_name}) ---")
        test_metrics = evaluate_model(best_model, X_test, y_test)
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Macro F1: {test_metrics['f1_macro']:.4f}")
        print(f"Test Weighted F1: {test_metrics['f1_weighted']:.4f}")
        print(f"Test Macro Precision: {test_metrics['precision_macro']:.4f}")
        print(f"Test Macro Recall: {test_metrics['recall_macro']:.4f}")
        
        # Per-class
        print("\nPer-class Test Results:")
        for cls in TARGET_NAMES:
            m = test_metrics['per_class'][cls]
            print(f"  {cls:10s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['support']}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = np.array(test_metrics['confusion_matrix'])
        header = "      " + "  ".join(f"{n[:4]:>4}" for n in TARGET_NAMES)
        print(header)
        for i, row in enumerate(cm):
            print(f"{TARGET_NAMES[i][:4]:>4} " + "  ".join(f"{v:>4}" for v in row))
        
        # Inference latency test
        n_samples = min(100, len(X_test))
        start = time.time()
        for _ in range(10):
            _ = best_model.predict(X_test[:n_samples])
        infer_time = (time.time() - start) / 10 / n_samples * 1000  # ms per sample
        print(f"\nInference latency: {infer_time:.3f} ms/sample")
        
        results["best_model"] = best_model_name
        results["test_metrics"] = test_metrics
        results["inference_ms"] = infer_time
    
    return results


def main():
    print("="*60)
    print("FACIAL EXPRESSION RECOGNITION BENCHMARK")
    print("="*60)
    
    # Load baseline data
    print("\nLoading baseline 48-feature dataset...")
    X_train_b, X_val_b, X_test_b, y_train_b, y_val_b, y_test_b, scaler_b, feat_names_b = load_baseline_data()
    
    # Load expanded data
    print("Loading expanded 1457-feature dataset...")
    X_train_e, X_val_e, X_test_e, y_train_e, y_val_e, y_test_e, scaler_e, feat_names_e = load_expanded_data()
    
    # Verify labels match
    assert np.array_equal(y_train_b, y_train_e), "Train labels mismatch!"
    assert np.array_equal(y_val_b, y_val_e), "Val labels mismatch!"
    assert np.array_equal(y_test_b, y_test_e), "Test labels mismatch!"
    print("[OK] Labels match across datasets")
    
    # Benchmark baseline
    baseline_results = benchmark_dataset(
        "BASELINE (48 engineered features)",
        X_train_b, X_val_b, X_test_b, y_train_b, y_val_b, y_test_b, feat_names_b
    )
    
    # Benchmark expanded
    expanded_results = benchmark_dataset(
        "EXPANDED (1457 MediaPipe features)",
        X_train_e, X_val_e, X_test_e, y_train_e, y_val_e, y_test_e, feat_names_e
    )
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Features':<10} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} {'Infer (ms)':<12}")
    print("-"*80)
    
    # Baseline models
    for model_name, res in baseline_results.items():
        if model_name in ["best_model", "test_metrics", "inference_ms"]:
            continue
        if "test_metrics" in baseline_results and baseline_results["best_model"] == model_name:
            tm = baseline_results["test_metrics"]
            print(f"{'Baseline ' + model_name:<20} {'48':<10} {tm['accuracy']:<10.4f} {tm['f1_macro']:<10.4f} {tm['f1_weighted']:<12.4f} {baseline_results.get('inference_ms', 0):<12.3f}")
    
    # Expanded models
    for model_name, res in expanded_results.items():
        if model_name in ["best_model", "test_metrics", "inference_ms"]:
            continue
        if "test_metrics" in expanded_results and expanded_results["best_model"] == model_name:
            tm = expanded_results["test_metrics"]
            print(f"{'Expanded ' + model_name:<20} {'1457':<10} {tm['accuracy']:<10.4f} {tm['f1_macro']:<10.4f} {tm['f1_weighted']:<12.4f} {expanded_results.get('inference_ms', 0):<12.3f}")
    
    # Save detailed results
    output = {
        "baseline": {k: v for k, v in baseline_results.items() if k != "model"},
        "expanded": {k: v for k, v in expanded_results.items() if k != "model"},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "train_samples": len(y_train_b),
            "val_samples": len(y_val_b),
            "test_samples": len(y_test_b),
            "classes": TARGET_NAMES
        }
    }
    
    # Remove model objects from output
    for k in ["baseline", "expanded"]:
        for model_name in list(output[k].keys()):
            if isinstance(output[k][model_name], dict) and "model" in output[k][model_name]:
                del output[k][model_name]["model"]
    
    with open('training/benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n[OK] Results saved to training/benchmark_results.json")
    
    return baseline_results, expanded_results


if __name__ == "__main__":
    baseline_results, expanded_results = main()