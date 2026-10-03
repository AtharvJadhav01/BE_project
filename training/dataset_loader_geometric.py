"""
Dataset loader for geometric facial features.

Loads and preprocesses FER2013 with geometric features.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
from sklearn.model_selection import train_test_split

from feature_extractor_geometric import GeometricFacialFeatures


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


def load_geometric_data():
    """Load geometric feature dataset."""
    print("Loading geometric feature dataset...")
    
    data_dir = Path('training/fer2013_features_geometric')
    
    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        meta = json.load(f)
    
    feature_names = meta['feature_names']
    
    print(f"Original feature count: {X_train.shape[1]}")
    from sklearn.model_selection import train_test_split
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # If validation set is empty, split training data
    if len(X_val) == 0:
        print("No validation set found. Splitting training data 90/10...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        print(f"New split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Clean features (remove constant/NaN)
    print("\nCleaning features (removing constant/NaN)...")
    print("\nCleaning features (removing constant/NaN)...")
    X_train, X_val, X_test, feature_names = clean_features(X_train, X_val, X_test, feature_names)
    print(f"Cleaned feature count: {X_train.shape[1]}")
    
    # Re-scale after cleaning (StandardScaler fitted on training only)
    from sklearn.preprocessing import StandardScaler
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    group_indices = get_feature_group_indices(feature_names)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices


def create_geometric_dataset():
    """Create geometric feature dataset from FER2013 images."""
    import cv2
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_extractor_geometric import create_geometric_extractor
    
    # Load FER2013 CSV
    import pandas as pd
    df = pd.read_csv('../data/fer2013.csv')
    
    # Split by usage
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PrivateTest']
    test_df = df[df['Usage'] == 'PublicTest']
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    extractor = create_geometric_extractor("../face_landmarker.task")
    
    def process_split(df, split_name):
        X_list = []
        y_list = []
        failed = 0
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"  {split_name}: {idx}/{len(df)}")
            
            # Parse pixels
            pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
            image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            
            features, detected = extractor.extract_features(image)
            if not detected or features is None:
                failed += 1
                continue
            
            X_list.append(features.to_array())
            y_list.append(row['emotion'])
        
        print(f"  {split_name}: {len(X_list)} successful, {failed} failed")
        return np.array(X_list), np.array(y_list)
    
    print("Processing training set...")
    X_train, y_train = process_split(train_df, "Train")
    print("Processing validation set...")
    X_val, y_val = process_split(val_df, "Val")
    print("Processing test set...")
    X_test, y_test = process_split(test_df, "Test")
    
    # Save
    data_dir = Path(__file__).parent / 'fer2013_features_geometric'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(data_dir / 'X_train.npy', np.array(X_train))
    np.save(data_dir / 'X_val.npy', np.array(X_val))
    np.save(data_dir / 'X_test.npy', np.array(X_test))
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'y_val.npy', y_val)
    np.save(data_dir / 'y_test.npy', y_test)
    
    # Save metadata
    feature_names = GeometricFacialFeatures.get_feature_names()
    meta = {
        'feature_names': feature_names,
        'feature_dim': len(feature_names),
        'label_map': {str(i): name for i, name in enumerate(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral'])},
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test)
    }
    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nDataset saved to {data_dir}/")
    print(f"Feature dimension: {len(feature_names)}")


if __name__ == "__main__":
    # create_geometric_dataset()  # Uncomment to generate dataset
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_geometric_data()
    print(f"Loaded geometric dataset: {X_train.shape[1]} features")