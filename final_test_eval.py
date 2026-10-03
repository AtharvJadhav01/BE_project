import time
import sys
sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()

# Best configuration: Landmarks + Blendshapes (1449 features)
indices = group_indices['landmarks'] + group_indices['blendshapes']

model = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000)
start = time.time()
model.fit(X_train[:, indices], y_train)
train_time = time.time() - start

y_pred = model.predict(X_test[:, indices])

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Per-class metrics
precision_per = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

# Inference time
n_samples = min(100, len(X_test))
start = time.time()
for _ in range(10):
    _ = model.predict(X_test[:n_samples, indices])
infer_time = (time.time() - start) / 10 / n_samples * 1000

print('='*60)
print('FINAL TEST EVALUATION - BEST CONFIGURATION')
print('='*60)
print('Configuration: LinearSVC + Landmarks + Blendshapes (1449 features)')
print('Params: C=0.1, class_weight=balanced, max_iter=10000')
print('')
print('Test Accuracy:          {:.4f} ({:.2f}%)'.format(accuracy, accuracy*100))
print('Test Macro Precision:   {:.4f}'.format(precision_macro))
print('Test Macro Recall:      {:.4f}'.format(recall_macro))
print('Test Macro F1:          {:.4f}'.format(f1_macro))
print('Test Weighted Precision: {:.4f}'.format(precision_weighted))
print('Test Weighted Recall:   {:.4f}'.format(recall_weighted))
print('Test Weighted F1:       {:.4f}'.format(f1_weighted))
print('')
print('Per-class Test Results:')
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
for i, cls in enumerate(class_names):
    support = int(np.sum(y_test == i))
    print('  {:10s}: P={:.4f} R={:.4f} F1={:.4f} support={}'.format(cls, precision_per[i], recall_per[i], f1_per[i], support))
print('')
print('Confusion Matrix (rows=true, cols=pred):')
header = '      ' + '  '.join('{:>4}'.format(n[:4]) for n in class_names)
print(header)
for i, row in enumerate(cm):
    print('{:>4} '.format(class_names[i][:4]) + '  '.join('{:>4}'.format(v) for v in row))
print('')
print('Training Time: {:.2f}s'.format(train_time))
print('Inference Latency: {:.3f} ms/sample'.format(infer_time))
print('')
print('COMPARISON WITH BASELINE (49.14% accuracy):')
print('  Baseline: 49.14% accuracy, 48.61% Macro F1, 49.12% Weighted F1')
print('  Optimized: {:.2f}% accuracy, {:.2f}% Macro F1, {:.2f}% Weighted F1'.format(accuracy*100, f1_macro*100, f1_weighted*100))
print('  Improvement: {:+.2f}pp accuracy, {:+.2f}pp Macro F1'.format((accuracy-0.4914)*100, (f1_macro-0.4861)*100))