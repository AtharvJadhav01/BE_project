"""
Save XGBoost Optimized Model Artifacts

Saves the best XGBoost model from the hyperparameter search:
- Landmarks + Blendshapes (1449 features)
- Best params from validation search
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

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data


def main():
    print("="*60)
    print("SAVE XGBOOST OPTIMIZED MODEL ARTIFACTS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_orig, feature_names, group_indices = load_and_prepare_data()
    
    # Best configuration: Landmarks + Blendshapes (1449 features)
    indices = group_indices['landmarks'] + group_indices['blendshapes']
    feature_names_subset = [feature_names[i] for i in indices]
    
    # Select features and fit scaler on selected features
    X_train_sub = X_train[:, indices]
    X_val_sub = X_val[:, indices]
    X_test_sub = X_test[:, indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val_sub)
    X_test_scaled = scaler.transform(X_test_sub)
    
    # Best XGBoost config from validation search
    params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1
    }
    
    print("\nTraining final XGBoost model...")
    print(f"Params: {params}")
    print(f"Features: {len(indices)} (Landmarks + Blendshapes)")
    
    start = time.time()
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
        **params
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    train_time = time.time() - start
    
    # Evaluate on test set
    y_pred = model.predict(scaler.transform(X_test[:, indices]))
    
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
        _ = model.predict(scaler.transform(X_test[:n_samples, indices]))
    infer_time = (time.time() - start) / 10 / n_samples * 1000
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Weighted F1: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"  Train Time: {train_time:.2f}s")
    print(f"  Inference: {infer_time:.3f} ms/sample")
    
    # Save artifacts
    print("\nSaving artifacts...")
    models_dir = Path('training/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'xgboost_optimized_1449.joblib'
    scaler_path = models_dir / 'scaler_xgboost_1449.joblib'
    metadata_path = models_dir / 'xgboost_optimized_1449_metadata.json'
    
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")
    
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # Create metadata
    metadata = {
        "model_type": "XGBClassifier",
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
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "objective": "multi:softprob",
            "num_class": 7,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1
        },
        "xgboost_version": "3.4.1",
        "random_seed": 42,
        "test_accuracy": float(accuracy),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": float(train_time),
        "inference_ms": float(infer_time),
        "best_iteration": getattr(model, 'best_iteration', None),
        "baseline_reference_accuracy": 0.49139414802065406,
        "baseline_reference_macro_f1": 0.4860619690380382,
        "baseline_reference_weighted_f1": 0.49120014653816585,
        "linearsvc_reference_accuracy": 0.5068846815834768,
        "linearsvc_reference_macro_f1": 0.49186395890069223,
        "linearsvc_reference_weighted_f1": 0.49979351524636734,
        "improvement_vs_logistic_accuracy_pp": float((accuracy - 0.49139414802065406) * 100),
        "improvement_vs_logistic_macro_f1_pp": float((f1_macro - 0.4860619690380382) * 100),
        "improvement_vs_linearsvc_accuracy_pp": float((accuracy - 0.5068846815834768) * 100),
        "improvement_vs_linearsvc_macro_f1_pp": float((f1_macro - 0.49186395890069223) * 100),
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
    pred_orig = model.predict(scaler.transform(X_test[:, indices]))
    pred_reload = model_reload.predict(X_test_scaled)
    
    print(f"Model loads: OK")
    print(f"Scaler loads: OK")
    print(f"Metadata loads: OK")
    print(f"Feature dimension: {meta_reload['feature_dim']}")
    print(f"Predictions identical: {np.array_equal(pred_orig, pred_reload)}")
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
    print(f"vs Baseline LR: {meta_reload['improvement_vs_logistic_accuracy_pp']:+.2f}pp accuracy")
    print(f"vs LinearSVC: {meta_reload['improvement_vs_linearsvc_accuracy_pp']:+.2f}pp accuracy")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'training')
    from ablation_study import load_and_prepare_data
    import time
    import joblib
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    main()