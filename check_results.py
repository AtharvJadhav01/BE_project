import json
with open('training/optimization_results.json') as f:
    data = json.load(f)
print('Experiments completed:', len(data['experiments']))
for exp in data['experiments'][-5:]:
    print(f"  {exp['model']}: {exp['params']} -> status={exp['status']}")
if data['best']:
    b = data['best']
    print('Best model:', b['model'])
    print('Best val macro F1:', b['val_macro_f1'])
    print('Best params:', b['params'])