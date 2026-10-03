import json
with open('training/optimization_results.json') as f:
    data = json.load(f)
models = {}
for exp in data['experiments']:
    m = exp['model']
    if m not in models:
        models[m] = {'completed': 0, 'errors': 0}
    if exp['status'] == 'COMPLETED':
        models[m]['completed'] += 1
    else:
        models[m]['errors'] += 1
for m, v in sorted(models.items()):
    print(f'{m}: completed={v["completed"]}, errors={v["errors"]}')
print('Total:', sum(v['completed'] for v in models.values()))