import json
with open('training/geometric_experiment_results.json') as f:
    data = json.load(f)
print('Experiments:', len(data['experiments']))
for exp in data['experiments']:
    if exp['status'] == 'COMPLETED':
        name = exp['name']
        val_f1 = exp['val_metrics']['f1_macro']
        test_f1 = exp['test_metrics']['f1_macro']
        acc = exp['test_metrics']['accuracy']
        print(f"{name:20s} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | Acc: {acc:.4f}")
if data.get('best'):
    b = data['best']
    print('\nBest:', b['name'], '- Val F1:', b['val_macro_f1'])