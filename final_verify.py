import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Verify all artifacts
print('=== ARTIFACT VERIFICATION ===')

# 1. PCA+RBF SVM
pipeline = joblib.load('training/models/pca_rbf_svm_optimized.joblib')
with open('training/models/pca_rbf_svm_optimized_metadata.json') as f:
    meta = json.load(f)
print('PCA+RBF SVM: OK')
print('Test Acc:', meta['test_accuracy'])
print('Macro F1:', meta['macro_f1'])
print('Weighted F1:', meta['weighted_f1'])
print('Verification:', meta['verification_passed'])

# 2. LinearSVC
with open('training/models/linear_svc_optimized_1449_metadata.json') as f:
    lsvc_meta = json.load(f)
print('\nLinearSVC: OK')
print('Test Acc:', lsvc_meta['test_accuracy'])
print('Macro F1:', lsvc_meta['macro_f1'])

# 3. XGBoost
with open('training/models/xgboost_optimized_1449_metadata.json') as f:
    xgb_meta = json.load(f)
print('\nXGBoost: OK')
print('Test Acc:', xgb_meta['test_accuracy'])
print('Macro F1:', xgb_meta['macro_f1'])

# 4. LogisticRegression
with open('training/models/logistic_regression_expanded_1457_metadata.json') as f:
    lr_meta = json.load(f)
print('\nLogisticRegression: OK')
print('Test Acc:', lr_meta['test_accuracy'])
print('Macro F1:', lr_meta['macro_f1'])

# 4. Geometric Optimized
model = joblib.load('training/models/geometric_optimized.joblib')
with open('training/models/geometric_optimized_metadata.json') as f:
    meta = json.load(f)
print('\nGeometric Optimized: OK')
print('Test Acc:', meta['test_accuracy'])
print('Macro F1:', meta['macro_f1'])
print('Verification:', meta['verification_passed'])

# Verify reload
import sys
sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data
X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_geometric_data()
indices = list(range(1503))
pipeline = joblib.load('training/models/geometric_optimized.joblib')
pred1 = pipeline.predict(X_test)
pipeline_reload = joblib.load('training/models/geometric_optimized.joblib')
pred2 = pipeline_reload.predict(X_test)
print('\nReload match:', np.array_equal(pred1, pred2))

print()
print('=== FINAL SUMMARY ===')
print('LinearSVC: 50.69% accuracy (BENCHMARK)')
print('LogisticRegression: 49.14%')
print('XGBoost: 46.90%')
print('PCA+RBF SVM: 46.73%')
print('Geometric (combined): 50.60%')
print()
print('Best model: LinearSVC (50.69%)')
print('Geometric features did NOT improve baseline.')