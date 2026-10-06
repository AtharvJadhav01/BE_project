"""
Webcam Expression Evaluation Tool

Evaluates a labeled webcam dataset against the CURRENT BEST VERIFIED MODEL.

Current best FER2013 benchmark (verified):
- Model: LinearSVC
- Features: 1,449 (Landmarks + Blendshapes)
- FER2013 PublicTest accuracy: 50.69%
- Macro F1: 49.19%
- Weighted F1: 49.98%

The verified LinearSVC model artifacts:
- Model: training/models/linear_svc_optimized_1449.joblib
- Scaler: training/models/scaler_optimized_1449.joblib
- Metadata: training/models/linear_svc_optimized_1449_metadata.json

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

# Add training to path for feature extractor
sys.path.insert(0, str(Path(__file__).parent))
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


class LinearSVCFeatureExtractor:
    """
    Feature extractor that produces exactly the 1449 features expected by the
    verified LinearSVC model (1404 landmarks + 48 blendshapes).
    
    This wraps the expanded feature extractor and selects only the features
    that the LinearSVC model was trained on.
    """
    
    def __init__(self, model_path: str, feature_names: List[str]):
        """
        Initialize with the MediaPipe model and target feature names.
        
        Args:
            model_path: Path to face_landmarker.task
            feature_names: List of 1449 feature names from LinearSVC metadata
        """
        self.extractor = ExpandedMediaPipeFeatureExtractor(model_path, normalize_landmarks=True)
        self.target_feature_names = feature_names
        
        # Map from target feature name to index in expanded extractor output
        self._build_feature_index_map()
    
    def _build_feature_index_map(self):
        """Build mapping from target feature names to indices in expanded extractor output."""
        # Get all feature names from expanded extractor
        expanded_names = ExpandedFacialFeatures.get_feature_names()
        
        # Build name-to-index mapping for expanded features
        expanded_name_to_idx = {name: i for i, name in enumerate(expanded_names)}
        
        # Build mapping for target features
        self.feature_indices = []
        missing = []
        for name in self.target_feature_names:
            if name in expanded_name_to_idx:
                self.feature_indices.append(expanded_name_to_idx[name])
            else:
                missing.append(name)
        
        if missing:
            raise ValueError(f"Missing features in expanded extractor: {missing}")
        
        print(f"Feature mapping: {len(self.target_feature_names)} target -> {len(self.feature_indices)} mapped")
        if len(self.target_feature_names) != len(self.feature_indices):
            raise ValueError(f"Feature count mismatch: target={len(self.target_feature_names)}, mapped={len(self.feature_indices)}")
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Extract features matching the LinearSVC expected format.
        
        Returns:
            Tuple of (1449-feature array or None, face_detected_flag)
        """
        features, face_detected = self.extractor.extract_features(image)
        if not face_detected or features is None:
            return None, False
        
        # Get full expanded feature vector
        expanded_array = features.to_array()
        
        # Select only the features needed by LinearSVC
        selected = expanded_array[self.feature_indices]
        
        return selected, True
    
    def close(self):
        self.extractor.close()


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
    """Evaluate the verified LinearSVC model on webcam dataset."""

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

    # Use verified LinearSVC model artifacts if not specified
    models_dir = Path(__file__).parent / "models"
    
    if model_path:
        mp = Path(model_path)
        sp = Path(scaler_path) if scaler_path else mp.parent / (mp.stem + "_scaler.joblib")
        mtp = Path(metadata_path) if metadata_path else mp.parent / (mp.stem + "_metadata.json")
    else:
        # Use verified LinearSVC artifacts
        mp = models_dir / "linear_svc_optimized_1449.joblib"
        sp = models_dir / "scaler_optimized_1449.joblib"
        mtp = models_dir / "linear_svc_optimized_1449_metadata.json"

    if not mp.exists():
        print("\n" + "=" * 60)
        print("LINEAR SVC MODEL ARTIFACT NOT FOUND")
        print("=" * 60)
        print(f"Expected: {mp}")
        print("\nAvailable models in training/models/:")
        for f in models_dir.glob("*.joblib"):
            print(f"  {f.name}")
        return {"status": "MODEL ARTIFACT NOT FOUND"}

    if not sp.exists():
        print(f"Scaler not found: {sp}")
        return {"status": "SCALER NOT FOUND"}

    if not mtp.exists():
        print(f"Metadata not found: {mtp}")
        return {"status": "METADATA NOT FOUND"}

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

    # Verify feature dimension matches LinearSVC
    feature_dim = metadata.get("feature_dim", 0)
    if feature_dim != 1449:
        print(f"Warning: Model feature_dim={feature_dim}, expected 1449")

    print(f"Model loaded: {metadata.get('model_type', 'unknown')}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Test accuracy (FER2013): {metadata.get('test_accuracy', 'N/A')}")

    # Get target feature names from metadata
    target_feature_names = metadata.get("feature_names", [])
    if not target_feature_names or len(target_feature_names) != 1449:
        print(f"Warning: Expected 1449 feature names in metadata, got {len(target_feature_names)}")
        # Fallback: construct from metadata feature_groups
        if "feature_group" in metadata and metadata["feature_group"] == "landmarks_blendshapes":
            print("Using landmarks_blendshapes feature group (1449 features)")

    # Initialize feature extractor with LinearSVC's expected features
    model_file = Path(__file__).parent.parent / "face_landmarker.task"
    if not model_file.exists():
        model_file = Path("face_landmarker.task")

    if not model_file.exists():
        print(f"MediaPipe model not found: {model_file}")
        return {"status": "MEDIAPIPE MODEL NOT FOUND"}

    # Initialize feature extractor with LinearSVC's expected feature names
    extractor = LinearSVCFeatureExtractor(str(model_file), target_feature_names)

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

        # Extract features matching LinearSVC expected format
        features, face_detected = extractor.extract_features(img)
        if not face_detected or features is None:
            failed += 1
            continue

        # Verify feature dimension
        if features.shape[0] != 1449:
            print(f"  Warning: Feature dimension mismatch: got {features.shape[0]}, expected 1449")
            failed += 1
            continue

        # Scale features
        try:
            feature_scaled = scaler.transform(features.reshape(1, -1))
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
    print("EVALUATION RESULTS (Webcam Dataset)")
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
    print(f"| FER2013 PublicTest | LinearSVC (1449ft) |  50.69%  |  49.19%  |    49.98%   |")
    print(f"| Webcam test set    | Same LinearSVC     | {accuracy*100:6.2f}% | {macro_f1*100:6.2f}%  |   {weighted_f1*100:6.2f}%   |")

    return {
        "status": "COMPLETED",
        "model": "LinearSVC (1449 features, landmarks + blendshapes)",
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
        "fer2013_benchmark": {
            "accuracy": 0.5069,
            "macro_f1": 0.4919,
            "weighted_f1": 0.4998
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate webcam expression dataset against verified LinearSVC model")
    parser.add_argument("--dataset", default="data/webcam_expression_dataset", help="Dataset directory")
    parser.add_argument("--model", help="Model path (optional, uses LinearSVC by default)")
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