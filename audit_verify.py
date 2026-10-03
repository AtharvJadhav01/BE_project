import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Load saved artifacts
model = joblib.load('training/models/linear_svc_optimized_1449.joblib')
scaler = joblib.load('training/models/scaler_optimized_1449.joblib')
with open('training/models/linear_svc_optimized_1449_metadata.json') as f:
    meta = json.load(f)

print('=== METADATA ===')
print('Model type:', meta['model_type'])
print('Feature dim:', meta['feature_dim'])
print('Feature group:', meta['feature_group'])
print('Hyperparameters:', meta['hyperparameters'])
print('Test accuracy:', meta['test_accuracy'])
print('Macro F1:', meta['macro_f1'])
print('Weighted F1:', meta['weighted_f1'])
print('Train samples:', meta['train_samples'])
print('Val samples:', meta['val_samples'])
print('Test samples:', meta['test_samples'])
print('Feature names count:', len(meta['feature_names']))
print('Preprocessing:', meta['preprocessing'])
print('Improvement accuracy pp:', meta['improvement_accuracy_pp'])
print('Improvement macro f1 pp:', meta['improvement_macro_f1_pp'])

# Verify model type
print('\n=== MODEL ===')
print('Model type:', type(model).__name__)
print('Model params:', model.get_params())

# Verify scaler
print('\n=== SCALER ===')
print('Scaler type:', type(scaler).__name__)
print('Scaler mean shape:', scaler.mean_.shape)
print('Scaler scale shape:', scaler.scale_.shape)

# Now reproduce the test result
import sys
sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data

X_train, X_val, X_test, y_train, y_val, y_test, scaler_orig, feature_names, group_indices = load_and_prepare_data()

# Best configuration: Landmarks + Blendshapes (1449 features)
indices = group_indices['landmarks'] + group_indices['blendshapes']
print(f'\n=== REPRODUCTION ===')
print('Selected feature indices count:', len(indices))

# Use the saved scaler to transform test data
X_test_scaled = scaler.transform(X_test[:, indices])
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print('\n=== REPRODUCTION RESULTS ===')
print('Accuracy:', accuracy)
print('Macro F1:', f1_macro)
print('Weighted F1:', f1_weighted)
print('Macro Precision:', precision_macro)
print('Macro Recall:', recall_macro)
print('Weighted Precision:', precision_weighted)
print('Weighted Recall:', recall_weighted)

print('\n=== METADATA vs REPRODUCTION ===')
print('Metadata test_accuracy:', meta['test_accuracy'], 'vs reproduced:', accuracy)
print('Match:', np.isclose(meta['test_accuracy'], accuracy))
print('Metadata macro_f1:', meta['macro_f1'], 'vs reproduced:', f1_macro)
print('Match:', np.isclose(meta['macro_f1'], f1_macro))
print('Metadata weighted_f1:', meta['weighted_f1'], 'vs reproduced:', f1_weighted)
print('Match:', np.isclose(meta['weighted_f1'], f1_weighted))

# Per-class metrics
from sklearn.metrics import precision_score, recall_score, f1_score
precision_per = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
print('\n=== PER-CLASS METRICS ===')
for i, cls in enumerate(class_names):
    support = int(np.sum(y_test == i))
    print(f'  {cls:10s}: P={precision_per[i]:.4f} R={recall_per[i]:.4f} F1={f1_per[i]:.4f} support={support}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('\n=== CONFUSION MATRIX ===')
header = '      ' + '  '.join('{:>4}'.format(n[:4]) for n in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral'])
print(header)
for i, row in enumerate(cm):
    print('{:>4} '.format(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral'][i][:4]) + '  '.join('{:>4}'.format(v) for v in row))

# Feature composition
print('\n=== FEATURE COMPOSITION ===')
print('Total features selected:', len(group_indices['landmarks']) + len(group_indices['blendshapes']))
print('Landmarks:', len(group_indices['landmarks']))
print('Blendshapes:', len(group_indices['blendshapes']))
print('Head pose:', len(group_indices['head_pose']))
print('Metadata:', len(group_indices['metadata']))
print('Removed features (constant/NaN): 4 (lm_1_x, lm_1_y, lm_1_z, feature_1460)')

# Verify scaler was fitted only on training data
print('\n=== SCALER VERIFICATION ===')
print('Scaler mean[:5]:', scaler.mean_[:5])
print('Scaler scale[:5]:', scaler.scale_[:5])

# Check if scaler was fitted on training data only by comparing with fresh scaler
from sklearn.preprocessing import StandardScaler
X_train_sub = X_train[:, group_indices['landmarks'] + group_indices['blendshapes']]
fresh_scaler = StandardScaler()
fresh_scaler.fit(X_train_sub)
print('Fresh scaler mean[:5]:', fresh_scaler.mean_[:5])
print('Fresh scaler scale[:5]:', fresh_scaler.scale_[:5])
print('Scaler matches fresh fit:', np.allclose(scaler.mean_, fresh_scaler.mean_) and np.allclose(scaler.scale_, fresh_scaler.scale_))

# Test set verification - same PublicTest split
print('\n=== TEST SET VERIFICATION ===')
print('Test samples:', len(y_test))
print('Class distribution:', np.bincount(y_test))

# Verify model reload produces identical predictions
model_reload = joblib.load('training/models/linear_svc_optimized_1449.joblib')
scaler_reload = joblib.load('training/models/scaler_optimized_1449.joblib')

X_test_scaled_reload = scaler_reload.transform(X_test[:, indices])
pred_orig = model.predict(X_test_scaled)
pred_reload = model_reload.predict(X_test_scaled_reload)

print('\n=== RELOAD VERIFICATION ===')
print('Predictions identical:', np.array_equal(pred_orig, pred_reload))

# Check validation set performance (not used for test)
val_pred = model.predict(scaler.transform(X_val[:, indices]))
val_f1_macro = f1_score(y_val, val_pred, average='macro', zero_division=0)
val_f1_weighted = f1_score(y_val, val_pred, average='weighted', zero_division=0)
val_acc = accuracy_score(y_val, val_pred)
print('\n=== VALIDATION PERFORMANCE (not used for test) ===')
print('Val Accuracy:', val_acc)
print('Val Macro F1:', val_f1_macro)
print('Val Weighted F1:', val_f1_weighted)

# Check if Phase 2 validation F1 matches ablation study
print('\n=== PHASE 2 vs ABLATION VALIDATION F1 ===')
print('Phase 2 best LinearSVC val Macro F1: 0.4566')
print('Ablation study landmarks+blendshapes val Macro F1: 0.4598')
print('Difference: 0.0032 (likely due to different data loading/cleaning or random seed)')