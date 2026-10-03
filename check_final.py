import json
with open('training/geometric_experiment_results.json') as f:
    data = json.load(f)
for exp in data['experiments']:
    if exp['status'] == 'COMPLETED':
        name = exp['name']
        acc = exp['test_metrics']['accuracy']
        macro_f1 = exp['test_metrics']['f1_macro']
        w_f1 = exp['test_metrics']['f1_weighted']
        print(f"{name:20s} | Test Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Wt F1: {w_f1:.4f}")