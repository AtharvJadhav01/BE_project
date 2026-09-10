import sys
import os
import numpy as np
import joblib
import json

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'training'))

from training.dataset_loader import FER2013_LABEL_MAP
from training.train_classifier import train_random_forest, train_svm, train_logistic_regression, evaluate_model

data_dir = r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work'
v2_dir = os.path.join(data_dir, 'training', 'fer2013_features_v2')

X_train = np.load(os.path.join(v2_dir, 'X_train.npy'))
X_val = np.load(os.path.join(v2_dir, 'X_val.npy'))
X_test = np.load(os.path.join(v2_dir, 'X_test.npy'))
y_train = np.load(os.path.join(v2_dir, 'y_train.npy'))
y_val = np.load(os.path.join(v2_dir, 'y_val.npy'))
y_test = np.load(os.path.join(v2_dir, 'y_test.npy'))

scaler = joblib.load(os.path.join(v2_dir, 'scaler.joblib'))
with open(os.path.join(v2_dir, 'metadata.json')) as f:
    metadata = json.load(f)

X_train_clean = np.nan_to_num(X_train, nan=0.0)
X_val_clean = np.nan_to_num(X_val, nan=0.0)
X_test_clean = np.nan_to_num(X_test, nan=0.0)

# Train RF
rf_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=15)
rf_test_metrics = evaluate_model(rf_model, X_test_clean, y_test)

# Train SVM
svm_model = train_svm(X_train_clean, y_train, C=10.0)
svm_test_metrics = evaluate_model(svm_model, X_test_clean, y_test)

# Train LR
lr_model = train_logistic_regression(X_train, y_train)
lr_test_metrics = evaluate_model(lr_model, X_test_clean, y_test)

# Print results
TARGET_NAMES = [FER2013_LABEL_MAP[i] for i in range(7)]

print('=== CLASSIFIER RESULTS ===')
print(f'Phase 2 Baseline: 43% accuracy (27 features, RF, FER2013)')
print()
print(f'RF Test Accuracy: {rf_test_metrics["accuracy"]:.4f} ({rf_test_metrics["accuracy"]*100:.1f}%)')
print(f'RF Test F1 (weighted): {rf_test_metrics["f1_weighted"]:.4f}')
print()
print(f'SVM Test Accuracy: {svm_test_metrics["accuracy"]:.4f} ({svm_test_metrics["accuracy"]*100:.1f}%)')
print(f'SVM Test F1 (weighted): {svm_test_metrics["f1_weighted"]:.4f}')
print()
print(f'LR Test Accuracy: {lr_test_metrics["accuracy"]:.4f} ({lr_test_metrics["accuracy"]*100:.1f}%)')
print(f'LR Test F1 (weighted): {lr_test_metrics["f1_weighted"]:.4f}')
print()

# Per-class details for RF
print('RF Per-class (test):')
for i, label in enumerate(TARGET_NAMES):
    m = rf_test_metrics["per_class"][label]
    print(f'  {label}: P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f}, support={m["support"]}')

print()
print('SVM Per-class (test):')
for i, label in enumerate(TARGET_NAMES):
    m = svm_test_metrics["per_class"][label]
    print(f'  {label}: P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f}, support={m["support"]}')

print()
print('LR Per-class (test):')
for i, label in enumerate(TARGET_NAMES):
    m = lr_test_metrics["per_class"][label]
    print(f'  {label}: P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f}, support={m["support"]}')

print()
print('NOTE: Dataset v2 metadata says 87 features but actual arrays have 48 columns.')
print('The current code\'s to_array() returns 37 features (verified separately).')
print('Results are with 48-dimensional vectors from the v2 artifacts.')