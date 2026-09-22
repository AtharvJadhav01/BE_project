"""
FER2013 Expanded Feature Extraction Pipeline

Extracts comprehensive MediaPipe features (landmarks, blendshapes, head pose)
from FER2013 dataset using the expanded feature extractor.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from training.feature_extractor_expanded import ExpandedMediaPipeFeatureExtractor, ExpandedFacialFeatures


# FER2013 label mapping
FER2013_LABEL_MAP = {
    0: "angry",
    1: "disgust", 
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral"
}

APP_LABEL_TO_IDX = {v: k for k, v in FER2013_LABEL_MAP.items()}


@dataclass
class DatasetStats:
    """Statistics about the extracted dataset."""
    total_samples: int
    face_detected: int
    no_face: int
    class_distribution: Dict[str, int]
    feature_dim: int
    train_samples: int
    val_samples: int
    test_samples: int


def load_fer2013_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate FER2013 CSV."""
    print(f"Loading FER2013 from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ["emotion", "pixels", "Usage"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Map emotions
    df["emotion_label"] = df["emotion"].map(FER2013_LABEL_MAP)
    
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['emotion_label'].value_counts()}")
    print(f"Split distribution:\n{df['Usage'].value_counts()}")
    
    return df


def pixels_to_image(pixels_str: str, size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """Convert FER2013 pixel string to grayscale image."""
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    if len(pixels) != size[0] * size[1]:
        raise ValueError(f"Expected {size[0]*size[1]} pixels, got {len(pixels)}")
    return pixels.reshape(size)


def extract_features_from_df(
    df: pd.DataFrame, 
    extractor: ExpandedMediaPipeFeatureExtractor,
    usage_filter: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Extract expanded MediaPipe features from FER2013 DataFrame.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        face_detected_indices: Indices where face was detected
        no_face_indices: Indices where no face was detected
    """
    import cv2
    
    if usage_filter:
        df = df[df["Usage"] == usage_filter].copy()
    
    if max_samples:
        df = df.head(max_samples).copy()
    
    print(f"Processing {len(df)} images...")
    
    X_list = []
    y_list = []
    face_detected_indices = []
    no_face_indices = []
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        try:
            # Convert pixels to image
            image = pixels_to_image(row["pixels"])
            
            # Convert grayscale to BGR for MediaPipe
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Extract features
            features, face_detected = extractor.extract_features(image_bgr)
            
            if face_detected and features is not None:
                X_list.append(features.to_array())
                y_list.append(APP_LABEL_TO_IDX[row["emotion_label"]])
                face_detected_indices.append(idx)
            else:
                no_face_indices.append(idx)
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            no_face_indices.append(idx)
        
        if len(X_list) % 500 == 0 and len(X_list) > 0:
            elapsed = time.time() - start_time
            rate = len(X_list) / elapsed
            print(f"  Processed {len(X_list)} faces... ({rate:.1f} faces/sec)")
    
    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, len(ExpandedFacialFeatures.get_feature_names())), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64) if y_list else np.empty(0, dtype=np.int64)
    
    elapsed = time.time() - start_time
    print(f"Face detected: {len(face_detected_indices)}, No face: {len(no_face_indices)}")
    print(f"Total time: {elapsed:.1f}s, Rate: {len(X_list)/elapsed:.1f} faces/sec")
    
    return X, y, face_detected_indices, no_face_indices


def prepare_dataset(
    data_dir: Path,
    model_path: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, DatasetStats]:
    """
    Complete dataset preparation pipeline with expanded features.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, stats
    """
    from sklearn.preprocessing import StandardScaler
    
    # Initialize extractor
    print(f"Initializing MediaPipe FaceLandmarker from {model_path}...")
    extractor = ExpandedMediaPipeFeatureExtractor(model_path, normalize_landmarks=True)
    
    try:
        # Load CSV
        csv_path = data_dir / "fer2013.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"FER2013 not found at {csv_path}")
        
        df = load_fer2013_csv(csv_path)
        
        # Extract features from training set
        print("\n=== Extracting features from Training set ===")
        X_train_raw, y_train_raw, train_face_idx, train_no_face_idx = extract_features_from_df(
            df, extractor, usage_filter="Training"
        )
        
        # Extract features from Public Test set
        print("\n=== Extracting features from Public Test set ===")
        X_test_raw, y_test_raw, test_face_idx, test_no_face_idx = extract_features_from_df(
            df, extractor, usage_filter="PublicTest"
        )
        
        # Extract features from Private Test set
        print("\n=== Extracting features from Private Test set ===")
        X_val_raw, y_val_raw, val_face_idx, val_no_face_idx = extract_features_from_df(
            df, extractor, usage_filter="PrivateTest"
        )
        
        # If PrivateTest is empty, split training data for validation
        if len(X_val_raw) == 0:
            print("\n=== PrivateTest empty, splitting Training data for validation ===")
            # Split training data into train/val
            val_size = int(len(X_train_raw) * val_split)
            indices = np.random.RandomState(random_state).permutation(len(X_train_raw))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            X_val_raw = X_train_raw[val_indices]
            y_val_raw = y_train_raw[val_indices]
            X_train_raw = X_train_raw[train_indices]
            y_train_raw = y_train_raw[train_indices]
            
            # Also split no-face indices proportionally
            val_no_face_idx = train_no_face_idx[:val_size]
            train_no_face_idx = train_no_face_idx[val_size:]
        
        # Combine all for statistics
        total_face = len(train_face_idx) + len(test_face_idx) + len(val_face_idx)
        total_no_face = len(train_no_face_idx) + len(test_no_face_idx) + len(val_no_face_idx)
        
        # Get class distribution
        all_labels = np.concatenate([y_train_raw, y_val_raw, y_test_raw])
        class_dist = {}
        for i, label in FER2013_LABEL_MAP.items():
            class_dist[label] = int(np.sum(all_labels == i))
        
        # Check for NaN/Inf in features
        for name, arr in [("X_train", X_train_raw), ("X_val", X_val_raw), ("X_test", X_test_raw)]:
            nan_count = np.sum(np.isnan(arr))
            inf_count = np.sum(np.isinf(arr))
            if nan_count > 0 or inf_count > 0:
                print(f"WARNING: {name} has {nan_count} NaN and {inf_count} Inf values")
                # Replace with 0
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features - fit ONLY on training data
        print("\n=== Scaling features (fit on train only) ===")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Verify no NaN after scaling
        for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            nan_count = np.sum(np.isnan(arr))
            inf_count = np.sum(np.isinf(arr))
            if nan_count > 0 or inf_count > 0:
                print(f"WARNING: {name} has {nan_count} NaN and {inf_count} Inf after scaling")
        
        # Create stats
        feature_names = ExpandedFacialFeatures.get_feature_names()
        stats = DatasetStats(
            total_samples=len(df),
            face_detected=total_face,
            no_face=total_no_face,
            class_distribution=class_dist,
            feature_dim=len(feature_names),
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test)
        )
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total samples: {stats.total_samples}")
        print(f"Face detected: {stats.face_detected} ({stats.face_detected/stats.total_samples*100:.1f}%)")
        print(f"No face: {stats.no_face} ({stats.no_face/stats.total_samples*100:.1f}%)")
        print(f"Feature dim: {stats.feature_dim}")
        print(f"Train: {stats.train_samples}, Val: {stats.val_samples}, Test: {stats.test_samples}")
        print(f"Class distribution: {stats.class_distribution}")
        
        return X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw, scaler, stats
        
    finally:
        extractor.close()


def save_dataset_artifacts(
    output_dir: Path,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scaler: Any,
    stats: DatasetStats,
    feature_names: List[str]
):
    """Save all dataset artifacts for training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature matrices
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)
    
    # Save scaler
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    # Save feature names and metadata
    metadata = {
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "label_map": FER2013_LABEL_MAP,
        "app_label_to_idx": APP_LABEL_TO_IDX,
        "stats": {
            "total_samples": stats.total_samples,
            "face_detected": stats.face_detected,
            "no_face": stats.no_face,
            "class_distribution": stats.class_distribution,
            "feature_dim": stats.feature_dim,
            "train_samples": stats.train_samples,
            "val_samples": stats.val_samples,
            "test_samples": stats.test_samples
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved dataset artifacts to {output_dir}")


def load_dataset_artifacts(data_dir: Path) -> Tuple:
    """Load saved dataset artifacts."""
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    scaler = joblib.load(data_dir / "scaler.joblib")
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FER2013 Expanded Feature Extraction Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/fer2013", help="Data directory")
    parser.add_argument("--model-path", type=str, default="face_landmarker.task", help="MediaPipe model path")
    parser.add_argument("--output-dir", type=str, default="training/fer2013_features_expanded", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split (for testing)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_path = args.model_path
    
    # Check if artifacts already exist
    if (output_dir / "X_train.npy").exists():
        print(f"Loading existing artifacts from {output_dir}")
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = load_dataset_artifacts(output_dir)
        print(f"Loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Feature dim: {metadata['feature_dim']}")
    else:
        print("Extracting expanded features from FER2013...")
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, stats = prepare_dataset(
            data_dir, model_path
        )
        
        save_dataset_artifacts(
            output_dir, X_train, X_val, X_test, y_train, y_val, y_test, scaler, stats,
            ExpandedFacialFeatures.get_feature_names()
        )
    
    print("\nExpanded dataset preparation complete!")