import sys
sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time

X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()

# Test with landmarks only
indices = group_indices['landmarks']
print(f'Testing landmarks only: {len(indices)} features')

model = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000)
start = time.time()
model.fit(X_train[:, group_indices['landmarks']], y_train)
train_time = time.time() - start

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision_macro = precision_score(y, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}

val_metrics = evaluate_model(model, X_val[:, group_indices['landmarks']], y_val)
test_metrics = evaluate_model(model, X_test[:, group_indices['landmarks']], y_test)
print('Landmarks only - Val Macro F1:', val_metrics['f1_macro'], 'Test Macro F1:', test_metrics['f1_macro'])