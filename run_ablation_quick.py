import sys
sys.path.insert(0, 'training')
from ablation_study import load_and_prepare_data
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time

X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision_macro = precision_score(y, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}

# Define subsets
subsets = {
    "full_1457": list(range(X_train.shape[1])),
    "landmarks_only": group_indices['landmarks'],
    "blendshapes_only": group_indices['blendshapes'],
    "head_pose_only": group_indices['head_pose'],
    "metadata_only": group_indices['metadata'],
    "landmarks_blendshapes": group_indices['landmarks'] + group_indices['blendshapes'],
    "landmarks_headpose": group_indices['landmarks'] + group_indices['head_pose'],
    "blendshapes_headpose": group_indices['blendshapes'] + group_indices['head_pose'],
    "landmarks_blendshapes_headpose": group_indices['landmarks'] + group_indices['blendshapes'] + group_indices['head_pose'],
    "no_landmarks": group_indices['blendshapes'] + group_indices['head_pose'] + group_indices['metadata'],
    "no_blendshapes": group_indices['landmarks'] + group_indices['head_pose'] + group_indices['metadata'],
    "no_headpose": group_indices['landmarks'] + group_indices['blendshapes'] + group_indices['metadata'],
}

model_params = {"C": 0.1, "class_weight": "balanced", "max_iter": 10000}

def evaluate_model(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision_macro = precision_score(y, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}

print("Running ablation study...")
print(f"{'Subset':30s} | {'N':4s} | {'Val F1':>8s} | {'Test F1':>8s}")
print("-" * 60)

for name, indices in subsets.items():
    if not indices:
        continue
    model = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000)
    start = time.time()
    model.fit(X_train[:, indices], y_train)
    train_time = time.time() - start
    
    val_metrics = evaluate_model(model, X_val[:, indices], y_val)
    test_metrics = evaluate_model(model, X_test[:, indices], y_test)
    
    print(f"{name:30s} | {len(indices):4d} | {val_metrics['f1_macro']:.4f} | {test_metrics['f1_macro']:.4f}")

print("Done!")