"""
Phase 5: XGBoost Experiment for Facial Expression Recognition

Tests whether XGBoost can outperform the current verified LinearSVC model
on the exact same FER2013 PublicTest evaluation protocol.

Current benchmarks:
- LogisticRegression (1457 features): 49.14% test accuracy
- LinearSVC (1449 features, Landmarks+Blendshapes): 50.69% test accuracy

XGBoost will use the same 1449-feature dataset (Landmarks + Blendshapes)
with the same FER2013 PublicTest evaluation protocol.
"""

import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

RESULTS_FILE = Path(__file__).parent / "xgboost_optimization_results.json"


def clean_features(X_train, X_val, X_test, feature_names):
    """Remove constant and NaN-only features - exact copy from ablation_study."""
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
            print(f"  Dropping feature {i}: {name} (NaN={nan_count}, std={std:.6f}")
    
    return X_train[:, keep_indices], X_val[:, keep_indices], X_test[:, keep_indices], keep_names


def load_and_prepare_data():
    """Load and prepare expanded data exactly as in benchmark/ablation."""
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
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Define feature group indices (after cleaning)
    group_indices = get_feature_group_indices(feature_names)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices


def get_feature_group_indices(feature_names):
    """Determine feature group indices after cleaning - exact copy from ablation_study."""
    groups = {
        "landmarks": [],
        "blendshapes": [],
        "head_pose": [],
        "metadata": []
    }
    
    for i, name in enumerate(feature_names):
        if name.startswith("lm_"):
            groups["landmarks"].append(i)
        elif name.startswith("bs_"):
            groups["blendshapes"].append(i)
        elif name.startswith("head_"):
            groups["head_pose"].append(i)
        elif name in ["inter_ocular_distance", "detection_confidence"]:
            groups["metadata"].append(i)
    
    print(f"Feature group sizes: { {k: len(v) for k, v in groups.items()} }")
    return groups


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
    
    precision_per = precision_score(y, y_pred, average=None, zero_division=0)
    recall_per = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y, y_pred)
    
    per_class = {}
    for i, name in enumerate(TARGET_NAMES):
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


def load_existing_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": [], "best": {}}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def experiment_already_done(results, model_name, params):
    """Check if this experiment was already completed."""
    for exp in results["experiments"]:
        if exp["model"] == model_name and exp["params"] == params:
            return True, exp
    return False, None


def run_xgboost_experiment(params, X_train, X_val, y_train, y_val, X_test, y_test, feature_names):
    """Run a single XGBoost experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: XGBoost with {params}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        # XGBoost params - early_stopping_rounds goes in constructor for XGBoost 3.x
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 7,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 50,
            **params
        }
        
        model = XGBClassifier(**xgb_params)
        
        # Use early stopping with validation set
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        train_time = time.time() - start
        
        # Evaluate on validation
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # Test metrics
        test_metrics = evaluate_model(model, X_test, y_test)
        
        # Inference time
        n_samples = min(100, len(X_test))
        start = time.time()
        for _ in range(10):
            _ = model.predict(X_test[:n_samples])
        infer_time = (time.time() - start) / 10 / n_samples * 1000
        
        result = {
            "model": "XGBoost",
            "params": params,
            "train_time": float(train_time),
            "inference_ms": float(infer_time),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_iteration": getattr(model, 'best_iteration', None),
            "status": "COMPLETED"
        }
        
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Inference: {infer_time:.3f} ms/sample")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
        print(f"  Best iteration: {result['best_iteration']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "model": "XGBoost",
            "params": params,
            "status": "ERROR",
            "error": str(e)
        }


def main():
    print("="*60)
    print("XGBOOST EXPERIMENT FOR FACIAL EXPRESSION RECOGNITION")
    print("="*60)
    print("Using FER2013 expanded features (1449: Landmarks + Blendshapes)")
    print("Train/Val only for model selection. Test set NEVER used.")
    print("Results saved incrementally to:", RESULTS_FILE)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()
    
    # Select best feature subset from ablation: Landmarks + Blendshapes (1449 features)
    indices = group_indices['landmarks'] + group_indices['blendshapes']
    print(f"\nUsing Landmarks + Blendshapes: {len(indices)} features")
    
    # Select features
    X_train_sub = X_train[:, indices]
    X_val_sub = X_val[:, indices]
    X_test_sub = X_test[:, indices]
    
    # Select feature names for this subset
    feature_names_sub = [feature_names[i] for i in indices]
    
    # Load existing results
    results = load_existing_results()
    print(f"Loaded {len(results['experiments'])} existing experiments")
    
    # Define XGBoost hyperparameter grid (conservative for small dataset)
    param_grids = [
        # Conservative configs first
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        # Lower learning rate with more trees
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
        # More regularization
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
    ]
    
    total_experiments = len(param_grids)
    print(f"\nTotal XGBoost experiments planned: {total_experiments}")
    
    completed = 0
    for params in param_grids:
        # Check if already done
        done, existing = experiment_already_done(results, "XGBoost", params)
        if done:
            print(f"  Skipping (already done): {params}")
            completed += 1
            continue
        
        result = run_xgboost_experiment(
            params, X_train, X_val, y_train, y_val, X_test, y_test, feature_names
        )
        
        results["experiments"].append(result)
        save_results(results)
        completed += 1
        
        # Update best
        if result["status"] == "COMPLETED":
            val_f1 = result["val_metrics"]["f1_macro"]
            if not results.get("best") or val_f1 > results["best"]["val_macro_f1"]:
                results["best"] = {
                    "model": "XGBoost",
                    "params": params,
                    "val_accuracy": result["val_metrics"]["accuracy"],
                    "val_macro_f1": val_f1,
                    "val_weighted_f1": result["val_metrics"]["f1_weighted"],
                    "train_time": result["train_time"],
                    "inference_ms": result["inference_ms"],
                    "best_iteration": result["best_iteration"]
                }
                save_results(results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("XGBOOST OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    
    if results.get("best"):
        b = results["best"]
        print(f"\nBEST XGBOOST CONFIG:")
        print(f"  Params: {b['params']}")
        print(f"  Val Accuracy: {b['val_accuracy']:.4f}")
        print(f"  Val Macro F1: {b['val_macro_f1']:.4f}")
        print(f"  Val Weighted F1: {b['val_weighted_f1']:.4f}")
        print(f"  Train Time: {b['train_time']:.2f}s")
        print(f"  Inference: {b['inference_ms']:.3f} ms")
        print(f"  Best Iteration: {b['best_iteration']}")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()