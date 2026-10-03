import json
with open('training/ablation_results.json') as f:
    data = json.load(f)
print('Experiments completed:', len(data['experiments']))
for exp in data['experiments']:
    print(f"  {exp['name']}: {exp['n_features']} feat | Val F1: {exp['val_metrics']['f1_macro']:.4f} | Test F1: {exp['test_metrics']['f1_macro']:.4f}")