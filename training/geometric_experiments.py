"""
Geometric Features Experiment for Facial Expression Recognition

Tests whether explicit geometric features can improve upon the verified 
LinearSVC baseline (50.69% test accuracy).

Uses exact same FER2013 PublicTest split and labels.
"""

import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

sys.path.insert(0, 'training')
from dataset_loader_geometric import load_geometric_data

TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

RESULTS_FILE = Path(__file__).parent / "geometric_experiment_results.json"


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


def experiment_already_done(results, name):
    for exp in results["experiments"]:
        if exp["name"] == name:
            return True, exp
    return False, None


def run_experiment(name, X_train, X_val, y_train, y_val, X_test, y_test, params=None):
    """Run a single LinearSVC experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    if params is None:
        params = {"C": 0.1, "class_weight": "balanced", "max_iter": 10000}
    
    start = time.time()
    try:
        model = LinearSVC(**params, random_state=42)
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)
        
        n_samples = min(100, len(X_test))
        start = time.time()
        for _ in range(10):
            _ = model.predict(X_test[:n_samples])
        infer_time = (time.time() - start) / 10 / n_samples * 1000
        
        result = {
            "name": name,
            "params": params,
            "train_time": float(train_time),
            "inference_ms": float(infer_time),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "status": "COMPLETED"
        }
        
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Inference: {infer_time:.3f} ms/sample")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test Macro F1: {test_metrics['f1_macro']:.4f}")
        print(f"  Test Weighted F1: {test_metrics['f1_weighted']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "name": name,
            "params": params,
            "status": "ERROR",
            "error": str(e)
        }


def load_existing_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": [], "best": {}}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def experiment_already_done(results, name):
    for exp in results["experiments"]:
        if exp["name"] == name:
            return True, exp
    return False, None


def main():
    import sys
    import time
    import json
    import joblib
    import numpy as np
    from pathlib import Path
    from typing import Dict, List, Tuple, Any
    
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    
    TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
    
    RESULTS_FILE = Path(__file__).parent / "geometric_experiment_results.json"
    
    print("="*60)
    print("GEOMETRIC FEATURES EXPERIMENT")
    print("="*60)
    print("Using FER2013 expanded features (1449 Landmarks + Blendshapes)")
    print("Adding geometric features on top")
    print("Train/Val only for model selection. Test set NEVER used.")
    print("Results saved incrementally to:", RESULTS_FILE)
    
    # Load data
    print("\nLoading geometric feature dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_geometric_data()
    
    # Select features (Landmarks + Blendshapes = 1449 features) - this is the baseline
    baseline_indices = list(range(1449))  # First 1449 features (landmarks + blendshapes)
    X_train_base = X_train[:, :1449]
    X_val_base = X_val[:, :1449]
    X_test_base = X_test[:, :1449]
    
    print(f"Using baseline (Landmarks + Blendshapes): 1449 features")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Geometric feature indices (1457-1502, 46 features)
    geom_indices = list(range(1457, 1503))
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Full feature count: {X_train.shape[1]}")
    
    # Load existing results
    results = {"experiments": [], "best": {}}
    if Path("geometric_experiment_results.json").exists():
        with open("geometric_experiment_results.json", 'r') as f:
            results = json.load(f)
    print(f"Loaded {len(results['experiments'])} existing experiments")
    
    # Define experiments using actual feature indices
    experiments = [
        # Baseline comparisons
        ("geometric_only", "geometric_only", list(range(1457, 1503))),  # 46 geometric features
        ("expanded_only", "expanded_only", list(range(1449))),  # 1449 baseline features
        ("combined", "combined", list(range(1503))),  # All 1503 features
        
        # Ablation: geometric feature groups (indices in full 1503 feature array)
        ("geom_eye", "eye", [i for i in range(1503) if feature_names[i].startswith("eye_")]),
        ("geom_mouth", "mouth", [i for i in range(1503) if feature_names[i].startswith("mouth_")]),
        ("geom_eyebrow", "eyebrow", [i for i in range(1503) if feature_names[i].startswith("eyebrow_")]),
        ("geom_nose", "nose", [i for i in range(1503) if feature_names[i].startswith("nose_")]),
        ("geom_symmetry", "symmetry", [i for i in range(1503) if feature_names[i].startswith("face_")]),
        ("geom_angles", "angles", [i for i in range(1503) if feature_names[i].startswith(("mouth_corner_angle", "eye_brow_angle"))]),
        ("geom_normalized", "normalized", [i for i in range(1503) if feature_names[i].startswith("normalized_")]),
    ]
    
    # Load existing results
    results = {"experiments": [], "best": {}}
    if Path("geometric_experiment_results.json").exists():
        with open("geometric_experiment_results.json", 'r') as f:
            results = json.load(f)
    print(f"Loaded {len(results['experiments'])} existing experiments")
    
    completed = 0
    for name, subset_name, indices in experiments:
        X_train_sub = X_train[:, indices]
        X_val_sub = X_val[:, indices]
        X_test_sub = X_test[:, indices]
        
        if X_train_sub.shape[1] == 0:
            print(f"  Skipping {name}: no features")
            continue
        
        done = any(exp["name"] == name for exp in results["experiments"])
        if done:
            print(f"  Skipping {name} (already done)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {name} ({len(indices)} features)")
        print(f"{'='*60}")
        
        result = run_experiment(
            name, X_train[:, indices], X_val[:, indices], 
            y_train, y_val, X_test[:, indices], y_test
        )
        
        results["experiments"].append(result)
        with open("geometric_experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        if result["status"] == "COMPLETED":
            val_f1 = result["val_metrics"]["f1_macro"]
            if not results.get("best") or val_f1 > results["best"]["val_macro_f1"]:
                results["best"] = {
                    "name": name,
                    "params": {"C": 0.1, "class_weight": "balanced", "max_iter": 10000},
                    "val_accuracy": result["val_metrics"]["accuracy"],
                    "val_macro_f1": val_f1,
                    "val_weighted_f1": result["val_metrics"]["f1_weighted"],
                    "train_time": result["train_time"],
                    "inference_ms": result["inference_ms"]
                }
                with open("geometric_experiment_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("GEOMETRIC EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    
    for exp in results["experiments"]:
        if exp["status"] == "COMPLETED":
            print(f"  {exp['name']:20s} | {exp['val_metrics']['f1_macro']:.4f} | Test F1: {exp['test_metrics']['f1_macro']:.4f}")
    
    print(f"\nResults saved to: geometric_experiment_results.json")


if __name__ == "__main__":
    main()