"""
Phase 2: Hyperparameter Optimization for Facial Expression Recognition

Resumable hyperparameter search on FER2013 expanded features (1457).
Uses ONLY train/validation data. Test set is NEVER used during optimization.

Results saved incrementally to training/optimization_results.json
"""

import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Try to import additional classifiers
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except ImportError:
    HAS_HGB = False
    print("HistGradientBoosting not available")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

# Label mapping
TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

# Results file
RESULTS_FILE = Path(__file__).parent / "optimization_results.json"


def clean_features(X_train, X_val, X_test, feature_names):
    """Remove constant and NaN-only features."""
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


def load_and_prepare_data():
    """Load and prepare expanded data exactly as in benchmark."""
    print("Loading expanded 1457-feature dataset...")
    
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
    
    # Re-scale after cleaning (StandardScaler fitted on training only)
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision_w = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y, y_pred, average="weighted", zero_division=0)
    
    precision_macro = precision_score(y, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }


def load_existing_results():
    """Load existing optimization results if they exist."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": [], "best": {}}


def save_results(results):
    """Save optimization results."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def experiment_already_done(results, model_name, params):
    """Check if this experiment was already completed."""
    for exp in results["experiments"]:
        if exp["model"] == model_name and exp["params"] == params:
            return True, exp
    return False, None


def run_experiment(model_name, params, X_train, X_val, y_train, y_val, X_test, y_test, feature_names):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {model_name} with {params}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        # Create model with params only - params already include all needed kwargs
        if model_name == "LogisticRegression":
            model = LogisticRegression(**params)
        elif model_name == "LinearSVC":
            model = LinearSVC(**params)
        elif model_name == "SVC":
            model = SVC(**params)
        elif model_name == "RandomForest":
            model = RandomForestClassifier(**params)
        elif model_name == "ExtraTrees":
            model = ExtraTreesClassifier(**params)
        elif model_name == "HistGradientBoosting":
            model = HistGradientBoostingClassifier(**params)
        elif model_name == "XGBoost":
            model = XGBClassifier(**params)
        else:
            return {"status": "ERROR", "error": f"Unknown model: {model_name}"}
        
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Evaluate on validation
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # Inference time
        n_samples = min(100, len(X_test))
        start = time.time()
        for _ in range(10):
            _ = model.predict(X_test[:n_samples])
        infer_time = (time.time() - start) / 10 / n_samples * 1000  # ms per sample
        
        result = {
            "model": model_name,
            "params": params,
            "train_time": float(train_time),
            "inference_ms": float(infer_time),
            "val_metrics": val_metrics,
            "status": "COMPLETED"
        }
        
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Inference: {infer_time:.3f} ms/sample")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "model": model_name,
            "params": params,
            "status": "ERROR",
            "error": str(e)
        }


def main():
    print("="*60)
    print("PHASE 2: HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print("Using FER2013 expanded features (1457 features)")
    print("Train/Val only for model selection. Test set NEVER used.")
    print("Results saved incrementally to:", RESULTS_FILE)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_and_prepare_data()
    
    # Load existing results
    results = load_existing_results()
    print(f"Loaded {len(results['experiments'])} existing experiments")
    
    # Define hyperparameter grids
    param_grids = {
        "LogisticRegression": [
            {"C": c, "max_iter": 2000, "random_state": 42, "class_weight": cw, "solver": "lbfgs", "n_jobs": -1}
            for c in [0.001, 0.01, 0.1, 1, 10, 100]
            for cw in [None, "balanced"]
        ],
        "LinearSVC": [
            {"C": c, "class_weight": cw, "max_iter": 10000}
            for c in [0.001, 0.01, 0.1, 1, 10, 100]
            for cw in [None, "balanced"]
        ],
        "SVC": [
            {"C": c, "kernel": "rbf", "gamma": g, "random_state": 42, "class_weight": cw}
            for c in [0.1, 1, 10]
            for g in ["scale", "auto", 0.01, 0.1]
            for cw in [None, "balanced"]
        ],
        "RandomForest": [
            {"n_estimators": n, "max_depth": d, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1, "class_weight": cw}
            for n in [100, 200, 300]
            for d in [None, 10, 15, 20]
            for cw in [None, "balanced"]
        ],
        "ExtraTrees": [
            {"n_estimators": n, "max_depth": d, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1, "class_weight": cw}
            for n in [100, 200, 300]
            for d in [None, 10, 15, 20]
            for cw in [None, "balanced"]
        ],
    }
    
    if HAS_HGB:
        param_grids["HistGradientBoosting"] = [
            {"max_iter": i, "max_depth": d, "learning_rate": lr, "random_state": 42}
            for i in [100, 200, 300]
            for d in [None, 5, 10]
            for lr in [0.01, 0.1]
        ]
    
    if HAS_XGB:
        param_grids["XGBoost"] = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr, "random_state": 42, "n_jobs": -1, "verbosity": 0}
            for n in [100, 200, 300]
            for d in [3, 5, 7]
            for lr in [0.01, 0.1]
        ]
    
    # Count total experiments
    total_experiments = sum(len(v) for v in param_grids.values())
    print(f"\nTotal experiments planned: {total_experiments}")
    
    completed = 0
    for model_name, param_list in param_grids.items():
        print(f"\n{'#'*60}")
        print(f"MODEL: {model_name} ({len(param_list)} configurations)")
        print(f"{'#'*60}")
        
        for params in param_list:
            # Check if already done
            done, existing = experiment_already_done(results, model_name, params)
            if done:
                print(f"  Skipping (already done): {params}")
                completed += 1
                continue
            
            result = run_experiment(
                model_name, params,
                X_train, X_val, y_train, y_val, X_test, y_test, feature_names
            )
            
            results["experiments"].append(result)
            save_results(results)
            completed += 1
            
            # Update best
            if result["status"] == "COMPLETED":
                val_f1 = result["val_metrics"]["f1_macro"]
                if not results["best"] or val_f1 > results["best"]["val_macro_f1"]:
                    results["best"] = {
                        "model": model_name,
                        "params": params,
                        "val_accuracy": result["val_metrics"]["accuracy"],
                        "val_macro_f1": val_f1,
                        "val_weighted_f1": result["val_metrics"]["f1_weighted"],
                        "train_time": result["train_time"],
                        "inference_ms": result["inference_ms"]
                    }
                    save_results(results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    
    if results["best"]:
        b = results["best"]
        print(f"\nBEST MODEL: {b['model']}")
        print(f"  Params: {b['params']}")
        print(f"  Val Accuracy: {b['val_accuracy']:.4f}")
        print(f"  Val Macro F1: {b['val_macro_f1']:.4f}")
        print(f"  Val Weighted F1: {b['val_weighted_f1']:.4f}")
        print(f"  Train Time: {b['train_time']:.2f}s")
        print(f"  Inference: {b['inference_ms']:.3f} ms")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()