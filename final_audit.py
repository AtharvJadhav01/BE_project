import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Verify all model artifacts load correctly
print('=== ARTIFACT VERIFICATION ===')

# 1. PCA + RBF SVM
pipeline = joblib.load('training/models/pca_rbf_svm_optimized.joblib')
with open('training/models/pca_rbf_svm_optimized_metadata.json') as f:
    meta = json.load(f)
print('PCA+RBF SVM: OK')
print('Test Accuracy:', meta['test_accuracy'])
print('Macro F1:', meta['macro_f1'])
print('Weighted F1:', meta['weighted_f1'])
print('Verification:', meta['verification_passed'])

# 2. LinearSVC (unchanged)
with open('training/models/linear_svc_optimized_1449_metadata.json') as f:
    lsvc_meta = json.load(f)
print('\nLinearSVC: OK')
print('Test Accuracy:', lsvc_meta['test_accuracy'])
print('Macro F1:', lsvc_meta['macro_f1'])

# 3. XGBoost (unchanged)
with open('training/models/xgboost_optimized_1449_metadata.json') as f:
    xgb_meta = json.load(f)
print('XGBoost: OK')
print('Test Accuracy:', xgb_meta['test_accuracy'])

# 4. LogisticRegression (unchanged)
with open('training/models/logistic_regression_expanded_1457_metadata.json') as f:
    lr_meta = json.load(f)
print('LogisticRegression: OK')
print('Test Accuracy:', lr_meta['test_accuracy'])

# 5. Verify reload works for PCA+RBF SVM
import sys
sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data
X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()
indices = group_indices['landmarks'] + group_indices['blendshapes']
pipeline_reload = joblib.load('training/models/pca_rbf_svm_optimized.joblib')
pred1 = pipeline.predict(X_test[:, indices])
pred2 = pipeline_reload.predict(X_test[:, indices])
print('\nReload predictions match:', np.array_equal(pred1, pred2))

print('\n=== FINAL SUMMARY ===')
print('PCA+RBF SVM: 46.73% accuracy, 45.72% Macro F1, 46.07% Weighted F1')
print('LinearSVC: 50.69% accuracy, 49.19% Macro F1, 49.98% Weighted F1')
print('XGBoost: 46.90% accuracy, 45.55% Macro F1, 45.87% Weighted F1')
print('LogisticRegression: 49.14% accuracy, 48.61% Macro F1, 49.12% Weighted F1')
print()
print('PCA+RBF SVM did NOT exceed LinearSVC (50.69%)')
print('Best model remains LinearSVC at 50.69% test accuracy')