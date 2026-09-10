"""
Facial Expression Classifier Training Script

Trains a lightweight classifier (Random Forest / SVM) on MediaPipe-extracted
FER2013 features for 7-class emotion classification.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from training.dataset_loader import load_dataset_artifacts, FER2013_LABEL_MAP
from training.feature_extractor import FacialFeatures


# Label mapping
IDX_TO_LABEL = {v: k for k, v in FER2013_LABEL_MAP.items()}
TARGET_NAMES = [FER2013_LABEL_MAP[i] for i in range(7)]


def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
        verbose=1
    )
    rf.fit(X_train, y_train)
    return rf


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str = "scale",
    random_state: int = 42
) -> Pipeline:
    """Train SVM with calibration for probability estimates."""
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state,
        class_weight="balanced",
        probability=False  # We'll use CalibratedClassifierCV
    )
    
    # Use calibrated classifier for probability estimates
    calibrated = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """Train Logistic Regression (baseline)."""
    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    return lr


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    target_names: List[str] = TARGET_NAMES
) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    y_pred = model.predict(X)
    
    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Classification report
    report = classification_report(y, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    
    # Per-class results
    per_class = {}
    for i, name in enumerate(target_names):
        per_class[name] = {
            "precision": float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
            "recall": float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
            "f1": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
            "support": int(np.sum(y == i))
        }
    
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "classification_report": report
    }


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "f1_weighted"
) -> Dict[str, float]:
    """Cross-validation evaluation."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    
    return {
        "cv_scores": scores.tolist(),
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores))
    }


def save_model(
    model: Any,
    scaler: Any,
    feature_names: List[str],
    label_map: Dict[int, str],
    output_dir: Path,
    metrics: Dict[str, Any],
    model_name: str = "emotion_classifier"
):
    """Save trained model with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # Save model
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "timestamp": timestamp,
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "label_map": {str(k): v for k, v in FER2013_LABEL_MAP.items()},
        "target_names": [FER2013_LABEL_MAP[i] for i in range(7)],
        "metrics": metrics,
        "feature_extraction": "MediaPipe FaceLandmarker (468 landmarks + 52 blendshapes)",
        "feature_count": 27,
        "notes": "Trained on FER2013 dataset using MediaPipe facial features"
    }
    
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return model_path


def load_model(model_path: Path):
    """Load trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(model_path.parent / "scaler.joblib")
    
    with open(model_path.parent / f"{model_path.stem}_metadata.json", "r") as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def predict_emotion(
    model: Any,
    scaler: Any,
    features: np.ndarray
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict emotion from feature vector.
    
    Returns:
        predicted_label, confidence, all_probabilities
    """
    # Ensure 2D array
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Predict
    pred_idx = model.predict(features_scaled)[0]
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features_scaled)[0]
        confidence = float(probas[pred_idx])
        all_probas = {FER2013_LABEL_MAP[i]: float(probas[i]) for i in range(7)}
    else:
        confidence = 1.0
        all_probas = {FER2013_LABEL_MAP[i]: 0.0 for i in range(7)}
        all_probas[FER2013_LABEL_MAP[pred_idx]] = 1.0
    
    predicted_label = FER2013_LABEL_MAP[pred_idx]
    
    return predicted_label, confidence, all_probas


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Facial Expression Classifier")
    parser.add_argument("--data-dir", type=str, default="training/fer2013_features", help="Dataset artifacts directory")
    parser.add_argument("--model-type", type=str, choices=["rf", "svm", "lr"], default="rf", help="Model type")
    parser.add_argument("--output-dir", type=str, default="training/models", help="Output directory")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    parser.add_argument("--rf-estimators", type=int, default=200, help="RF n_estimators")
    parser.add_argument("--rf-max-depth", type=int, default=15, help="RF max_depth")
    parser.add_argument("--svm-C", type=float, default=10.0, help="SVM C parameter")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Load dataset
    print(f"Loading dataset from {data_dir}")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = load_dataset_artifacts(data_dir)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Feature dim: {X_train.shape[1]}")
    
    # Train model
    print(f"\nTraining {args.model_type} model...")
    
    if args.model_type == "rf":
        model = train_random_forest(
            X_train, y_train,
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth
        )
        model_name = f"rf_emotion_{args.rf_estimators}est_{args.rf_max_depth}depth"
    elif args.model_type == "svm":
        model = train_svm(X_train, y_train, C=args.svm_C)
        model_name = f"svm_emotion_C{args.svm_C}"
    elif args.model_type == "lr":
        model = train_logistic_regression(X_train, y_train)
        model_name = "lr_emotion"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Cross-validation if requested
    if args.cv:
        print("\nRunning cross-validation...")
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)
        print(f"CV F1-weighted: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
    
    # Evaluate on validation set
    print("\n=== Validation Set Evaluation ===")
    val_metrics = evaluate_model(model, X_val, y_val)
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"Precision (weighted): {val_metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {val_metrics['recall_weighted']:.4f}")
    
    print("\nPer-class Performance:")
    for label, metrics in val_metrics["per_class"].items():
        print(f"  {label}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, support={metrics['support']}")
    
    print("\nConfusion Matrix:")
    cm = np.array(val_metrics["confusion_matrix"])
    print("      " + "  ".join(f"{n[:4]:>4}" for n in TARGET_NAMES))
    for i, row in enumerate(cm):
        print(f"{TARGET_NAMES[i][:4]:>4} " + "  ".join(f"{v:>4}" for v in row))
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    
    print("\nPer-class Performance:")
    for label, metrics in test_metrics["per_class"].items():
        print(f"  {label}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(test_metrics["confusion_matrix"])
    print("      " + "  ".join(f"{n[:4]:>4}" for n in TARGET_NAMES))
    for i, row in enumerate(cm):
        print(f"{TARGET_NAMES[i][:4]:>4} " + "  ".join(f"{v:>4}" for v in row))
    
    # Save model
    output_dir = Path(args.output_dir)
    save_model(
        model=model,
        scaler=scaler,
        feature_names=FacialFeatures.get_feature_names(),
        label_map=FER2013_LABEL_MAP,
        output_dir=output_dir,
        metrics={
            "validation": val_metrics,
            "test": test_metrics,
            "model_params": {
                "model_type": args.model_type,
                "rf_estimators": args.rf_estimators if args.model_type == "rf" else None,
                "rf_max_depth": args.rf_max_depth if args.model_type == "rf" else None,
                "svm_C": args.svm_C if args.model_type == "svm" else None
            }
        },
        model_name=model_name
    )
    
    print(f"\n✅ Training complete! Model saved to {output_dir}/{model_name}.joblib")
    
    # Quick inference test
    print("\n=== Inference Test ===")
    sample_idx = 0
    pred_label, confidence, probas = predict_emotion(model, scaler, X_test[sample_idx])
    true_label = IDX_TO_LABEL[y_test[sample_idx]]
    print(f"Sample {sample_idx}: True={true_label}, Pred={pred_label}, Confidence={confidence:.4f}")


if __name__ == "__main__":
    import cv2  # Import here to avoid issues
    main()