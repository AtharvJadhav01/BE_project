import sys
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data

X_train, X_val, X_test, y_train, y_val, y_test, scaler_orig, feature_names, group_indices = load_and_prepare_data()

# Best XGBoost config
indices = group_indices['landmarks'] + group_indices['blendshapes']

# Need to fit scaler on selected features only
X_train_sub = X_train[:, indices]
X_val_sub = X_val[:, indices]
X_test_sub = X_test[:, indices]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sub)
X_val_scaled = scaler.transform(X_val_sub)
X_test_scaled = scaler.transform(X_test_sub)

params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1}

model = XGBClassifier(
    objective='multi:softprob',
    num_class=7,
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
    **params
)

model.fit(X_train_scaled, y_train, eval_set=[(X_val_sub, y_val)], verbose=False)

# Test evaluation
y_pred = model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

precision_per = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print('=== XGBOOST FINAL TEST RESULTS ===')
print('Test Accuracy:', accuracy, '({:.2f}%)'.format(accuracy*100))
print('Macro F1:', macro_f1, '({:.2f}%)'.format(macro_f1*100))
print('Weighted F1:', weighted_f1, '({:.2f}%)'.format(weighted_f1*100))
print('Macro Precision:', precision_macro)
print('Macro Recall:', recall_macro)
print('Weighted Precision:', precision_weighted)
print('Weighted Recall:', recall_weighted)
print('')

print('Per-class Test Results:')
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
for i, cls in enumerate(class_names):
    support = int(np.sum(y_test == i))
    print('  {:10s}: P={:.4f} R={:.4f} F1={:.4f} support={}'.format(cls, precision_per[i], recall_per[i], f1_per[i], support))

print('')
print('Confusion Matrix (rows=true, cols=pred):')
header = '      ' + '  '.join('{:>4}'.format(n[:4]) for n in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral'])
print(header)
for i, row in enumerate(cm):
    print('{:>4} '.format(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral'][i][:4]) + '  '.join('{:>4}'.format(v) for v in row))

print('')
print('COMPARISON WITH BENCHMARKS:')
print('| Model              | Features | Test Accuracy | Macro F1 | Weighted F1 |')
print('| ------------------ | -------: | ------------: | -------: | ----------: |')
print('| LogisticRegression |     1457 |        49.14% |   48.61% |      49.12% |')
print('| LinearSVC          |     1449 |        50.69% |   49.19% |      49.98% |')
print('| XGBoost            |     1449 |       {:6.2f}% |   {:6.2f}%  |    {:6.2f}%   |'.format(accuracy*100, macro_f1*100, weighted_f1*100))

print('')
print('XGBoost vs LinearSVC: {:+.2f}pp accuracy, {:+.2f}pp Macro F1'.format((accuracy - 0.5068846815834768)*100, (macro_f1 - 0.49186395890069223)*100))