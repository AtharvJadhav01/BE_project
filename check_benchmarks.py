import json

with open('training/benchmark_results.json') as f:
    data = json.load(f)

# Print expanded results
exp = data['expanded']
print('=== EXPANDED BENCHMARK RESULTS ===')
for model_name in ['RandomForest', 'SVM', 'LogisticRegression', 'MLP']:
    if model_name in exp:
        m = exp[model_name]
        if 'val_metrics' in m:
            v = m['val_metrics']
            print(f'{model_name}: val_acc={v["accuracy"]:.4f}, val_macro_f1={v["f1_macro"]:.4f}, val_weighted_f1={v["f1_weighted"]:.4f}')

# Check test metrics
if 'test_metrics' in exp:
    tm = exp['test_metrics']
    print(f'\nBest model: {exp.get("best_model", "?")}')
    print(f'Test Accuracy: {tm["accuracy"]:.4f}')
    print(f'Test Macro F1: {tm["f1_macro"]:.4f}')
    print(f'Test Weighted F1: {tm["f1_weighted"]:.4f}')

# Baseline results
base = data['baseline']
print('\n=== BASELINE BENCHMARK RESULTS ===')
for model_name in ['RandomForest', 'SVM', 'LogisticRegression', 'MLP']:
    if model_name in base:
        m = base[model_name]
        if 'val_metrics' in m:
            v = m['val_metrics']
            print(f'{model_name}: val_acc={v["accuracy"]:.4f}, val_macro_f1={v["f1_macro"]:.4f}, val_weighted_f1={v["f1_weighted"]:.4f}')

if 'test_metrics' in base:
    tm = base['test_metrics']
    print(f'\nBest model: {base.get("best_model", "?")}')
    print(f'Test Accuracy: {tm["accuracy"]:.4f}')
    print(f'Test Macro F1: {tm["f1_macro"]:.4f}')
    print(f'Test Weighted F1: {tm["f1_weighted"]:.4f}')