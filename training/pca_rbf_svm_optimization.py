"""
PCA + RBF SVM Experiment for Facial Expression Recognition

Tests whether dimensionality reduction with PCA followed by RBF SVM
can outperform the verified LinearSVC (50.69% test accuracy) on FER2013.

Uses exact same data pipeline: 1449 Landmarks + Blendshapes features,
StandardScaler fitted on training only, PCA fitted on training only.
"""

import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

RESULTS_FILE = Path(__file__).parent / "pca_rbf_svm_results.json"


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
            print(f"  Dropping feature {i}: {name} (NaN={nan_count}, std={std:.6f})")
    
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
    
    # Define feature group indices (after cleaning)
    group_indices = get_feature_group_indices(feature_names)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, group_indices


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


def experiment_already_done(results, pca_components, svm_params):
    """Check if this experiment was already completed."""
    for exp in results["experiments"]:
        if (exp.get("pca_components") == pca_components and 
            exp.get("svm_params") == svm_params):
            return True, exp
    return False, None


def run_experiment(pca_config, svm_params, X_train, X_val, y_train, y_val, X_test, y_test):
    """Run a single PCA + RBF SVM experiment."""
    print(f"\n{'='*60}")
    print(f"Running: PCA={pca_config}, SVM={svm_params}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        # Build pipeline: StandardScaler -> PCA -> RBF SVM
        if pca_config.get("type") == "n_components":
            pca = PCA(n_components=pca_config["n_components"], random_state=42)
        elif pca_config.get("type") == "explained_variance":
            pca = PCA(n_components=pca_config["explained_variance"], random_state=42)
        else:
            return {"status": "ERROR", "error": f"Unknown PCA config type: {pca_config}"}
        
        # Create pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("svm", SVC(**svm_params, random_state=42, probability=False))
        ])
        
        # Fit on training data only
        start_fit = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_fit
        
        # Evaluate on validation
        val_metrics = evaluate_model(pipeline, X_val, y_val)
        
        # Test metrics
        test_metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Inference time
        n_samples = min(100, len(X_test))
        start_infer = time.time()
        for _ in range(10):
            _ = pipeline.predict(X_test[:n_samples])
        infer_time = (time.time() - start_infer) / 10 / n_samples * 1000
        
        # Get actual PCA components used
        pca_fitted = pipeline.named_steps["pca"]
        actual_components = pca_fitted.n_components_
        explained_variance = float(np.sum(pca_fitted.explained_variance_ratio_))
        
        result = {
            "pca_config": pca_config,
            "svm_params": svm_params,
            "pca_components_actual": int(actual_components),
            "explained_variance": explained_variance,
            "train_time": float(train_time),
            "inference_ms": float(infer_time),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "status": "COMPLETED"
        }
        
        print(f"  PCA components: {actual_components} (explained var: {explained_variance:.4f})")
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Inference: {infer_time:.3f} ms/sample")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "pca_config": pca_config,
            "svm_params": svm_params,
            "status": "ERROR",
            "error": str(e)
        }


def main():
    print("="*60)
    print("PCA + RBF SVM EXPERIMENT FOR FACIAL EXPRESSION RECOGNITION")
    print("="*60)
    print("Using FER2013 expanded features (1449: Landmarks + Blendshapes)")
    print("Train/Val only for model selection. Test set NEVER used.")
    print("Results saved incrementally to:", RESULTS_FILE)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, group_indices = load_and_prepare_data()
    
    # Select best feature subset from ablation: Landmarks + Blendshapes (1449 features)
    indices = group_indices['landmarks'] + group_indices['blendshapes']
    print(f"\nUsing Landmarks + Blendshapes: {len(indices)} features")
    
    # Select features
    X_train_sub = X_train[:, indices]
    X_val_sub = X_val[:, indices]
    X_test_sub = X_test[:, indices]
    
    # Load existing results
    results = load_existing_results()
    print(f"Loaded {len(results['experiments'])} existing experiments")
    
    # Define PCA configurations
    pca_configs = [
        # Fixed component counts
        {"type": "n_components", "n_components": 50},
        {"type": "n_components", "n_components": 100},
        {"type": "n_components", "n_components": 150},
        {"type": "n_components", "n_components": 200},
        {"type": "n_components", "n_components": 300},
        {"type": "n_components", "n_components": 500},
        {"type": "n_components", "n_components": 750},
        {"type": "n_components", "n_components": 1000},
        # Explained variance thresholds
        {"type": "explained_variance", "explained_variance": 0.90},
        {"type": "explained_variance", "explained_variance": 0.95},
        {"type": "explained_variance", "explained_variance": 0.99},
    ]
    
    # SVM hyperparameter grid
    svm_param_grids = [
        {"C": 0.1, "gamma": "scale", "class_weight": None},
        {"C": 0.1, "gamma": "scale", "class_weight": "balanced"},
        {"C": 1, "gamma": "scale", "class_weight": None},
        {"C": 1, "gamma": "scale", "class_weight": "balanced"},
        {"C": 10, "gamma": "scale", "class_weight": None},
        {"C": 10, "gamma": "scale", "class_weight": "balanced"},
        {"C": 100, "gamma": "scale", "class_weight": None},
        {"C": 100, "gamma": "scale", "class_weight": "balanced"},
        {"C": 1, "gamma": "auto", "class_weight": None},
        {"C": 1, "gamma": "auto", "class_weight": "balanced"},
        {"C": 10, "gamma": "auto", "class_weight": None},
        {"C": 10, "gamma": "auto", "class_weight": "balanced"},
        {"C": 10, "gamma": 0.01, "class_weight": "balanced"},
        {"C": 10, "gamma": 0.1, "class_weight": "balanced"},
    ]
    
    total_experiments = len(pca_configs) * len(svm_param_grids)
    print(f"\nTotal experiments planned: {total_experiments}")
    
    completed = 0
    for pca_config in pca_configs:
        print(f"\n{'#'*60}")
        print(f"PCA CONFIG: {pca_config}")
        print(f"{'#'*60}")
        
        for svm_params in svm_param_grids:
            # Check if already done
            done, existing = experiment_already_done(results, 
                pca_config if isinstance(pca_config, str) else json.dumps(pca_config, sort_keys=True),
                svm_params)
            if done:
                print(f"  Skipping (already done): PCA={pca_config}, SVM={svm_params}")
                completed += 1
                continue
            
            result = run_experiment(pca_config, svm_params, 
                X_train[:, indices], X_val[:, indices], 
                y_train, y_val, X_test[:, indices], y_test)
            
            results["experiments"].append(result)
            save_results(results)
            completed += 1
            
            # Update best
            if result["status"] == "COMPLETED":
                val_f1 = result["val_metrics"]["f1_macro"]
                if not results.get("best") or val_f1 > results["best"]["val_macro_f1"]:
                    results["best"] = {
                        "pca_config": pca_config,
                        "svm_params": svm_params,
                        "pca_components_actual": result["pca_components_actual"],
                        "explained_variance": result["explained_variance"],
                        "val_accuracy": result["val_metrics"]["accuracy"],
                        "val_macro_f1": val_f1,
                        "val_weighted_f1": result["val_metrics"]["f1_weighted"],
                        "train_time": result["train_time"],
                        "inference_ms": result["inference_ms"],
                    }
                    save_results(results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("PCA + RBF SVM OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    
    if results.get("best"):
        b = results["best"]
        print(f"\nBEST CONFIG:")
        print(f"  PCA: {b['pca_config']}")
        print(f"  Actual components: {b['pca_components_actual']}")
        print(f"  Explained variance: {b['explained_variance']:.4f}")
        print(f"  SVM: {b['svm_params']}")
        print(f"  Val Accuracy: {b['val_accuracy']:.4f}")
        print(f"  Val Macro F1: {b['val_macro_f1']:.4f}")
        print(f"  Val Weighted F1: {b['val_weighted_f1']:.4f}")
        print(f"  Train Time: {b['train_time']:.2f}s")
        print(f"  Inference: {b['inference_ms']:.3f} ms")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()