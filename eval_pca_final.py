import sys
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data

X_train, X_val, X_test, y_train, y_val, y_test, scaler_orig, feature_names, group_indices = load_and_prepare_data()

# Best config: PCA 100 components, SVM C=10, gamma='scale', class_weight=None
indices = group_indices['landmarks'] + group_indices['blendshapes']

pca = PCA(n_components=100, random_state=42)
svm = SVC(C=10, gamma='scale', class_weight=None, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', pca),
    ('svm', svm)
])

pipeline.fit(X_train[:, indices], y_train)

y_pred = pipeline.predict(X_test[:, indices])

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

print('=== FINAL TEST EVALUATION ===')
print('Accuracy:', accuracy, '({:.2f}%)'.format(accuracy*100))
print('Macro F1:', macro_f1, '({:.2f}%)'.format(macro_f1*100))
print('Weighted F1:', weighted_f1, '({:.2f}%)'.format(weighted_f1*100))
print('Macro Precision:', precision_score(y_test, y_pred, average='macro', zero_division=0))
print('Macro Recall:', recall_score(y_test, y_pred, average='macro', zero_division=0))

# Per-class
prec = precision_score(y_test, y_pred, average=None, zero_division=0)
rec = recall_score(y_test, y_pred, average=None, zero_division=0)
f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
for i, cls in enumerate(classes):
    support = int(np.sum(y_test == i))
    print('{:10s}: P={:.4f} R={:.4f} F1={:.4f} support={}'.format(cls, prec[i], rec[i], f1[i], support))

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(cm)

print('\nCOMPARISON:')
print('LinearSVC: 50.69%')
print('XGBoost: 46.90%')
print('PCA+RBF SVM: {:.2f}%'.format(accuracy*100))
print('Diff vs LinearSVC: {:+.2f}pp'.format((accuracy-0.5068846815834768)*100))