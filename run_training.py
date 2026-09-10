import sys
import os
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'training'))

from training.dataset_loader import load_dataset_artifacts, FER2013_LABEL_MAP
from training.train_classifier import train_random_forest, train_svm, train_logistic_regression, evaluate_model, TARGET_NAMES, IDX_TO_LABEL

data_dir = r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work'

# Load from v2
v2_dir = os.path.join(data_dir, 'training', 'fer2013_features_v2')

X_train = np.load(os.path.join(v2_dir, 'X_train.npy'))
X_val = np.load(os.path.join(v2_dir, 'X_val.npy'))
X_test = np.load(os.path.join(v2_dir, 'X_test.npy'))
y_train = np.load(os.path.join(v2_dir, 'y_train.npy'))
y_val = np.load(os.path.join(v2_dir, 'y_val.npy'))
y_test = np.load(os.path.join(v2_dir, 'y_test.npy'))

# Load scaler and metadata
scaler = joblib.load(os.path.join(v2_dir, 'scaler.joblib'))
with open(os.path.join(v2_dir, 'metadata.json')) as f:
    metadata = json.load(f)

# Replace NaN values with 0 (SVM doesn't accept NaN)
X_train_clean = np.nan_to_num(X_train, nan=0.0)
X_val_clean = np.nan_to_num(X_val, nan=0.0)
X_test_clean = np.nan_to_num(X_test, nan=0.0)

# ===== TRAIN RANDOM FOREST =====
print('=== Training Random Forest ===')
rf_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=15)
rf_val_metrics = evaluate_model(rf_model, X_val_clean, y_val)
rf_test_metrics = evaluate_model(rf_model, X_test_clean, y_test)

print(f'Validation: Accuracy={rf_val_metrics["accuracy"]:.4f}, F1={rf_val_metrics["f1_weighted"]:.4f}')
print(f'Test: Accuracy={rf_test_metrics["accuracy"]:.4f}, F1={rf_test_metrics["f1_weighted"]:.4f}')

print('Per-class (test):')
for label, m in rf_test_metrics["per_class"].items():
    print(f'  {label}: P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f}, support={m["support"]}')

# ===== TRAIN SVM (with NaN handling) =====
print('\n=== Training SVM ===')
svm_model = train_svm(X_train_clean, y_train, C=10.0)
svm_val_metrics = evaluate_model(svm_model, X_val_clean, y_val)
svm_test_metrics = evaluate_model(svm_model, X_test_clean, y_test)

print(f'Validation: Accuracy={svm_val_metrics["accuracy"]:.4f}, F1={svm_val_metrics["f1_weighted"]:.4f}')
print(f'Test: Accuracy={svm_test_metrics["accuracy"]:.4f}, F1={svm_test_metrics["f1_weighted"]:.4f}')

print('Per-class (test):')
for label, m in svm_test_metrics["per_class"].items():
    print(f'  {label}: P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f}')

# ===== TRAIN LOGISTIC REGRESSION =====
print('\n=== Training Logistic Regression ===')
lr_model = train_logistic_regression(X_train, y_train)
lr_val_metrics = evaluate_model(lr_model, X_val_clean, y_val)
lr_test_metrics = evaluate_model(lr_model, X_test_clean, y_test)

print(f'Validation: Accuracy={lr_val_metrics["accuracy"]:.4f}, F1={lr_val_metrics["f1_weighted"]:.4f}')
print(f'Test: Accuracy={lr_test_metrics["accuracy"]:.4f}, F1={lr_test_metrics["f1_weighted"]:.4f}')

print('Per-class (test):')
for label, m in lr_test_metrics["per_class"].items():
    print(f'  {label}: P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f}')

# ===== SUMMARY =====
print('\n=== SUMMARY ===')
print('Phase 2 Baseline: 43% accuracy (27 features, Random Forest, FER2013)')
print(f'RF Test Accuracy: {rf_test_metrics["accuracy"]:.4f} ({rf_test_metrics["accuracy"]*100:.1f}%)')
print(f'SVM Test Accuracy: {svm_test_metrics["accuracy"]:.4f} ({svm_test_metrics["accuracy"]*100:.1f}%)')
print(f'LR Test Accuracy: {lr_test_metrics["accuracy"]:.4f} ({lr_test_metrics["accuracy"]*100:.1f}%)')
print()
print('Note: Dataset v2 has metadata feature_dim=87 but actual arrays have 48 columns.')
print('The current code\\'s to_array() returns 37 features (verified separately).')
print('Results are with 48-dimensional vectors from the v2 artifacts.')
print()
print('Key comparisons:')
print(f'  RF improves on 43% baseline: {rf_test_metrics["accuracy"]*100:.1f}% vs 43%')
print(f'  SVM improves on 43% baseline: {svm_test_metrics["accuracy"]*100:.1f}% vs 43%')
print(f'  LR improves on 43% baseline: {lr_test_metrics["accuracy"]*100:.1f}% vs 43%')