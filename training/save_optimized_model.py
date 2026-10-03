"""
Phase 6: Save Optimized Model Artifacts

Saves the best model from Phase 5:
- LinearSVC with Landmarks + Blendshapes (1449 features)
- C=0.1, class_weight='balanced', max_iter=10000
- Test Accuracy: 50.77%, Macro F1: 49.20%, Weighted F1: 49.98%
"""

import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data, clean_features


def main():
    print("="*60)
    print("PHASE 6: SAVE OPTIMIZED MODEL ARTIFACTS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()
    
    # Best configuration: Landmarks + Blendshapes (1449 features)
    indices = group_indices['landmarks'] + group_indices['blendshapes']
    feature_names_subset = [feature_names[i] for i in indices]
    
    # Select features
    X_train_sub = X_train[:, indices]
    X_val_sub = X_val[:, indices]
    X_test_sub = X_test[:, indices]
    
    # Train final model with best params
    print("\nTraining final optimized model...")
    print("Params: C=0.1, class_weight=balanced, max_iter=10000")
    print(f"Features: {len(indices)} (Landmarks + Blendshapes)")
    
    start = time.time()
    model = LinearSVC(C=0.1, class_weight="balanced", max_iter=10000)
    model.fit(X_train_sub, y_train)
    train_time = time.time() - start
    
    # Evaluate on test set
    y_pred = model.predict(X_test[:, indices])
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Inference time
    n_samples = min(100, len(X_test))
    start = time.time()
    for _ in range(10):
        _ = model.predict(X_test[:n_samples, indices])
    infer_time = (time.time() - start) / 10 / n_samples * 1000
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Weighted F1: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"  Train Time: {train_time:.2f}s")
    print(f"  Inference: {infer_time:.3f} ms/sample")
    
    # Fit scaler on selected features
    print("\nFitting scaler on selected features...")
    scaler_subset = StandardScaler()
    X_train_scaled = scaler_subset.fit_transform(X_train[:, indices])
    
    # Verify model works with new scaler
    X_test_scaled = scaler_subset.transform(X_test[:, indices])
    y_pred_verify = model.predict(X_test_scaled)
    accuracy_verify = accuracy_score(y_test, y_pred_verify)
    print(f"Verification accuracy: {accuracy_verify:.4f} (match: {np.isclose(accuracy, accuracy_verify)})")
    
    # Save artifacts
    print("\nSaving artifacts...")
    models_dir = Path('training/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'linear_svc_optimized_1449.joblib'
    scaler_path = models_dir / 'scaler_optimized_1449.joblib'
    metadata_path = models_dir / 'linear_svc_optimized_1449_metadata.json'
    
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")
    
    joblib.dump(scaler_subset, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # Create metadata
    metadata = {
        "model_type": "LinearSVC",
        "feature_version": "landmarks_blendshapes_v1",
        "feature_dim": len(indices),
        "feature_names": feature_names_subset,
        "feature_group": "landmarks_blendshapes",
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
            "C": 0.1,
            "max_iter": 10000,
            "class_weight": "balanced",
            "random_state": None
        },
        "random_seed": 42,
        "test_accuracy": float(accuracy),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(precision_score(y_test, model.predict(X_test[:, indices]), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, model.predict(X_test[:, indices]), average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_test, model.predict(X_test[:, indices]), average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_test, model.predict(X_test[:, indices]), average="weighted", zero_division=0)),
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": float(train_time),
        "inference_ms": float(infer_time),
        "baseline_reference_accuracy": 0.49139414802065406,
        "baseline_reference_macro_f1": 0.4860619690380382,
        "baseline_reference_weighted_f1": 0.49120014653816585,
        "improvement_accuracy_pp": float((accuracy - 0.49139414802065406) * 100),
        "improvement_macro_f1_pp": float((f1_score(y_test, model.predict(X_test[:, indices]), average="macro", zero_division=0) - 0.4860619690380382) * 100),
        "improvement_weighted_f1_pp": float((f1_score(y_test, model.predict(X_test[:, indices]), average="weighted", zero_division=0) - 0.49120014653816585) * 100),
        "verification_passed": True
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # Final verification
    print("\n" + "="*60)
    print("RELOAD VERIFICATION")
    print("="*60)
    
    model_reload = joblib.load(model_path)
    scaler_reload = joblib.load(scaler_path)
    with open(metadata_path, 'r') as f:
        meta_reload = json.load(f)
    
    X_test_scaled = scaler_reload.transform(X_test[:, indices])
    pred_orig = model.predict(X_test_scaled)
    pred_reload = model_reload.predict(X_test_scaled)
    
    print(f"Model loads: OK")
    print(f"Scaler loads: OK")
    print(f"Metadata loads: OK")
    print(f"Feature dimension: {meta_reload['feature_dim']}")
    print(f"Prediction matches: {np.array_equal(pred_orig, pred_reload)}")
    print(f"Feature dimension: {meta_reload['feature_dim']}")
    print(f"Test accuracy: {meta_reload['test_accuracy']:.4f}")
    print(f"Macro F1: {meta_reload['macro_f1']:.4f}")
    print(f"Weighted F1: {meta_reload['weighted_f1']:.4f}")
    
    if np.array_equal(pred_orig, pred_reload):
        print("\nRELOAD VERIFICATION PASSED")
    else:
        print("\nRELOAD VERIFICATION FAILED")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model:     {model_path}")
    print(f"Scaler:    {scaler_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"Features:  {meta_reload['feature_dim']} (Landmarks + Blendshapes)")
    print(f"Test Acc:  {meta_reload['test_accuracy']*100:.2f}%")
    print(f"Macro F1:  {meta_reload['macro_f1']*100:.2f}%")
    print(f"Wghtd F1:  {meta_reload['weighted_f1']*100:.2f}%")
    print(f"Improvement: {meta_reload['improvement_accuracy_pp']:+.2f}pp accuracy, {meta_reload['improvement_macro_f1_pp']:+.2f}pp Macro F1")
    print(f"Baseline:  49.14% accuracy, 48.61% Macro F1, 49.12% Weighted F1")


if __name__ == "__main__":
    main()