"""
Webcam Expression Evaluation Tool

Evaluates a labeled webcam dataset against the CURRENT BEST MODEL.

Current best FER2013 benchmark:
- Model: LogisticRegression
- Features: 1,457 expanded MediaPipe features
- FER2013 PublicTest accuracy: 49.14%

IMPORTANT: The 1,457-feature LogisticRegression model artifact was NOT saved
during the benchmark phase. This tool will report that and stop if the artifact
is not available.

Usage:
    python training/evaluate_webcam.py --dataset data/webcam_expression_dataset
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from face.detector import FaceDetector, FaceDetectionResult
from training.feature_extractor_expanded import (
    ExpandedMediaPipeFeatureExtractor,
    ExpandedFacialFeatures,
)


EXPRESSION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
    "neutral",
]

EXPRESSION_TO_IDX = {cls: i for i, cls in enumerate(EXPRESSION_CLASSES)}


def find_best_model_artifact() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find the 1,457-feature LogisticRegression model artifacts."""
    models_dir = Path(__file__).parent / "models"

    # Expected artifact names for the 1,457-feature LogisticRegression model
    # Based on the benchmark phase naming convention
    possible_names = [
        "lr_emotion_1457feat.joblib",
        "logistic_regression_1457.joblib",
        "lr_expanded_1457.joblib",
        "best_logistic_regression_1457.joblib",
    ]

    for name in possible_names:
        model_path = models_dir / name
        scaler_path = models_dir / name.replace(".joblib", "_scaler.joblib")
        metadata_path = models_dir / name.replace(".joblib", "_metadata.json")
        if model_path.exists():
            return model_path, scaler_path, metadata_path

    # Also check for any LogisticRegression model with 1457 features
    for model_file in models_dir.glob("*.joblib"):
        if "lr" in model_file.name.lower() or "logistic" in model_file.name.lower():
            # Check metadata for feature count
            meta_file = models_dir / model_file.name.replace(".joblib", "_metadata.json")
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                    if meta.get("feature_dim") == 1457:
                        scaler_path = models_dir / model_file.name.replace(".joblib", "_scaler.joblib")
                        return model_file, scaler_path, meta_file
                except Exception:
                    pass

    return None, None, None


def load_webcam_dataset(dataset_dir: Path) -> Tuple[List[str], List[str]]:
    """Load webcam dataset from directory structure."""
    image_paths = []
    labels = []

    for cls in EXPRESSION_CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            for img_path in cls_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(cls)

    return image_paths, labels


def evaluate_on_webcam(
    dataset_dir: str = "data/webcam_expression_dataset",
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Dict:
    """Evaluate the best model on webcam dataset."""

    print("=== Webcam Expression Evaluation ===\n")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Dataset directory does not exist: {dataset_path}")
        return {"status": "DATASET NOT FOUND"}

    # Load dataset
    image_paths, labels = load_webcam_dataset(dataset_path)
    if not image_paths:
        print("No images found in dataset")
        return {"status": "DATASET EMPTY"}

    print(f"Loaded {len(image_paths)} images from {len(set(labels))} classes")
    label_counts = Counter(labels)
    for cls in EXPRESSION_CLASSES:
        print(f"  {cls}: {label_counts.get(cls, 0)}")

    # Find or use specified model
    if model_path:
        mp = Path(model_path)
        sp = Path(scaler_path) if scaler_path else mp.parent / (mp.stem + "_scaler.joblib")
        mtp = Path(metadata_path) if metadata_path else mp.parent / (mp.stem + "_metadata.json")
    else:
        mp, sp, mtp = find_best_model_artifact()

    if mp is None or not mp.exists():
        print("\n" + "=" * 60)
        print("BEST MODEL ARTIFACT NOT AVAILABLE")
        print("=" * 60)
        print("The 1,457-feature LogisticRegression model that achieved")
        print("49.14% on FER2013 PublicTest was never saved to disk.")
        print("")
        print("Available models in training/models/:")
        for f in (Path(__file__).parent / "models").glob("*.joblib"):
            print(f"  {f.name}")
        print("")
        print("To proceed, you must:")
        print("  1. Retrain the LogisticRegression on 1,457 expanded features")
        print("  2. Save the model, scaler, and metadata")
        print("  3. Re-run this evaluation")
        return {"status": "BEST MODEL ARTIFACT NOT AVAILABLE"}

    print(f"\nLoading model: {mp}")
    print(f"Loading scaler: {sp}")
    print(f"Loading metadata: {mtp}")

    # Load artifacts
    try:
        model = joblib.load(mp)
        scaler = joblib.load(sp)
        with open(mtp, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return {"status": "MODEL LOAD FAILED", "error": str(e)}

    # Verify feature dimension
    feature_dim = metadata.get("feature_dim", 0)
    if feature_dim != 1457:
        print(f"Warning: Model feature_dim={feature_dim}, expected 1457")

    print(f"Model loaded: {metadata.get('model_type', 'unknown')}")
    print(f"Feature dimension: {feature_dim}")

    # Initialize feature extractor
    model_file = Path(__file__).parent.parent / "face_landmarker.task"
    if not model_file.exists():
        # Try to find it
        model_file = Path("face_landmarker.task")

    if not model_file.exists():
        print(f"MediaPipe model not found: {model_file}")
        return {"status": "MEDIAPIPE MODEL NOT FOUND"}

    extractor = ExpandedMediaPipeFeatureExtractor(str(model_file), normalize_landmarks=True)
    detector = FaceDetector({
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    })

    # Process images and extract features
    print("\nExtracting features from webcam images...")
    X = []
    y_true = []
    failed = 0

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        if i % 20 == 0 and i > 0:
            print(f"  Processed {i}/{len(image_paths)}...")

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        # Extract expanded features
        features, face_detected = extractor.extract_features(img)
        if not face_detected or features is None:
            failed += 1
            continue

        # Convert to array and scale
        feature_array = features.to_array().reshape(1, -1)
        if feature_array.shape[1] != feature_dim:
            # Feature dimension mismatch
            failed += 1
            continue

        try:
            feature_scaled = scaler.transform(feature_array)
            X.append(feature_scaled[0])
            y_true.append(EXPRESSION_TO_IDX[label])
        except Exception as e:
            failed += 1
            continue

    print(f"Successfully processed: {len(X)} images")
    print(f"Failed: {failed} images")

    if len(X) == 0:
        print("No valid samples for evaluation")
        return {"status": "NO VALID SAMPLES"}

    X = np.array(X)
    y_true = np.array(y_true)

    # Predict
    print("\nRunning predictions...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    per_class = {}
    for i, cls in enumerate(EXPRESSION_CLASSES):
        if cls in label_counts:
            cls_mask = (y_true == i)
            if np.any(cls_mask):
                cls_precision = precision_score(y_true == i, y_pred == i, zero_division=0)
                cls_recall = recall_score(y_true == i, y_pred == i, zero_division=0)
                cls_f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
                per_class[cls] = {
                    "precision": float(cls_precision),
                    "recall": float(cls_recall),
                    "f1": float(cls_f1),
                    "support": int(label_counts[cls]),
                }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EXPRESSION_CLASSES)))

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:   {weighted_recall:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    print("\nPer-class metrics:")
    for cls in EXPRESSION_CLASSES:
        if cls in per_class:
            m = per_class[cls]
            print(f"  {cls:10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
        else:
            print(f"  {cls:10s}: N/A (no samples)")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c[:4]:>4s}" for c in EXPRESSION_CLASSES))
    for i, cls in enumerate(EXPRESSION_CLASSES):
        row = "  ".join(f"{cm[i, j]:4d}" for j in range(len(EXPRESSION_CLASSES)))
        print(f"{cls[:4]:>4s}: {row}")

    print("=" * 60)

    # Comparison table
    print("\nCOMPARISON WITH FER2013 BENCHMARK:")
    print("| Dataset            | Model              | Accuracy | Macro F1 | Weighted F1 |")
    print("| ------------------ | ------------------ | -------: | -------: | ----------: |")
    print(f"| FER2013 PublicTest | LogisticRegression |  49.14%  |  48.61%  |    49.12%   |")
    print(f"| Webcam test set    | Same model         | {accuracy*100:6.2f}% | {macro_f1*100:6.2f}%  |   {weighted_f1*100:6.2f}%   |")

    return {
        "status": "COMPLETED",
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(X),
        "failed_samples": failed,
        "class_distribution": dict(label_counts),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate webcam expression dataset against best model")
    parser.add_argument("--dataset", default="data/webcam_expression_dataset", help="Dataset directory")
    parser.add_argument("--model", help="Model path (optional, auto-discover if not provided)")
    parser.add_argument("--scaler", help="Scaler path (optional)")
    parser.add_argument("--metadata", help="Metadata path (optional)")
    args = parser.parse_args()

    result = evaluate_on_webcam(
        dataset_dir=args.dataset,
        model_path=args.model,
        scaler_path=args.scaler,
        metadata_path=args.metadata,
    )

    # Save results
    output_path = Path(args.dataset) / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()