"""
Phase 4: Ablation Study - Feature Set Comparison

Compares different feature subsets using the best model from Phase 2 (LinearSVC).
Tests feature importance by training on subsets of the 1457 expanded features.

Feature groups in expanded 1457 features:
- Landmarks (1404): 468 landmarks × 3 (x, y, z)
- Blendshapes (52): MediaPipe blendshape coefficients
- Head pose (3): pitch, yaw, roll
- Metadata (2): inter_ocular_distance, detection_confidence
"""

import sys
import time
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

RESULTS_FILE = Path(__file__).parent / "ablation_results.json"


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
    """Load and prepare expanded data with feature groups."""
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
    
    # Re-scale after cleaning
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Define feature group indices (after cleaning)
    # We need to map original feature groups to cleaned indices
    group_indices = get_feature_group_indices(feature_names)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices


def get_feature_group_indices(feature_names):
    """Determine feature group indices after cleaning."""
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
    
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }


def train_and_evaluate(model, X_train, X_val, y_train, y_val, X_test, y_test):
    """Train model and evaluate on validation and test."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Validation metrics
    val_metrics = evaluate_model(model, X_val, y_val)
    
    # Test metrics
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Inference time
    n_samples = min(100, len(X_test))
    start = time.time()
    for _ in range(10):
        _ = model.predict(X_test[:n_samples])
    infer_time = (time.time() - start) / 10 / n_samples * 1000
    
    return {
        "train_time": float(time.time() - start),
        "inference_ms": float(infer_time),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def load_existing_results():
    if Path(__file__).parent.joinpath("ablation_results.json").exists():
        with open("ablation_results.json", 'r') as f:
            return json.load(f)
    return {"experiments": []}


def save_results(results):
    with open("ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)


def run_ablation_experiments():
    """Run ablation experiments with different feature subsets."""
    print("="*60)
    print("PHASE 4: ABLATION STUDY - FEATURE SET COMPARISON")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()
    
    # Define feature subsets to test
    subsets = {
        "full_1457": list(range(len(feature_names))),
        "landmarks_only": group_indices["landmarks"],
        "blendshapes_only": group_indices["blendshapes"],
        "head_pose_only": group_indices["head_pose"],
        "metadata_only": group_indices["metadata"],
        "landmarks_blendshapes": group_indices["landmarks"] + group_indices["blendshapes"],
        "landmarks_headpose": group_indices["landmarks"] + group_indices["head_pose"],
        "blendshapes_headpose": group_indices["blendshapes"] + group_indices["head_pose"],
        "landmarks_blendshapes_headpose": group_indices["landmarks"] + group_indices["blendshapes"] + group_indices["head_pose"],
        "no_landmarks": group_indices["blendshapes"] + group_indices["head_pose"] + group_indices["metadata"],
        "no_blendshapes": group_indices["landmarks"] + group_indices["head_pose"] + group_indices["metadata"],
        "no_headpose": group_indices["landmarks"] + group_indices["blendshapes"] + group_indices["metadata"],
    }
    
    # Best model from Phase 2: LinearSVC with C=0.1, class_weight='balanced'
    best_params = {"C": 0.1, "class_weight": "balanced", "max_iter": 10000}
    
    # Load existing results
    results = {"experiments": []}
    if Path("ablation_results.json").exists():
        with open("ablation_results.json", 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results['experiments'])} existing experiments")
    
    def experiment_done(name):
        for exp in results["experiments"]:
            if exp["name"] == name:
                return True
        return False
    
    for subset_name, indices in subsets.items():
        if experiment_done(subset_name):
            print(f"  Skipping {subset_name} (already done)")
            continue
        
        if not indices:
            print(f"  Skipping {subset_name}: no features")
            continue
        
        print(f"\n{'='*60}")
        print(f"ABLATION: {subset_name} ({len(indices)} features)")
        print(f"{'='*60}")
        
        # Select features
        X_train_sub = X_train[:, indices]
        X_val_sub = X_val[:, indices]
        X_test_sub = X_test[:, indices]
        
        # Train LinearSVC with best params
        model = LinearSVC(C=0.1, class_weight="balanced", max_iter=10000)
        
        result = train_and_evaluate(
            model, X_train[:, indices], X_val[:, indices],
            y_train, y_val, X_test[:, indices], y_test
        )
        
        result["name"] = subset_name
        result["n_features"] = len(indices)
        result["feature_indices"] = indices
        
        print(f"  Val Accuracy: {result['val_metrics']['accuracy']:.4f}")
        print(f"  Val Macro F1: {result['val_metrics']['f1_macro']:.4f}")
        print(f"  Val Weighted F1: {result['val_metrics']['f1_weighted']:.4f}")
        print(f"  Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Test Macro F1: {result['test_metrics']['f1_macro']:.4f}")
        print(f"  Test Weighted F1: {result['test_metrics']['f1_weighted']:.4f}")
        
        results["experiments"].append({
            "name": subset_name,
            "n_features": len(indices),
            "params": {"C": 0.1, "class_weight": "balanced", "max_iter": 10000},
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "train_time": result["train_time"],
            "inference_ms": result["inference_ms"],
            "status": "COMPLETED"
        })
        
        # Save incrementally
        with open("ablation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    
    for exp in results["experiments"]:
        print(f"  {exp['name']:30s} | {exp['n_features']:4d} feat | Val F1: {exp['val_metrics']['f1_macro']:.4f} | Test F1: {exp['test_metrics']['f1_macro']:.4f}")
    
    return results


def main():
    results = run_ablation_experiments()
    
    # Save final results
    with open("ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ablation_results.json")


if __name__ == "__main__":
    main()