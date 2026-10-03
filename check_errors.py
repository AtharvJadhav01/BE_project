import json
with open('training/optimization_results.json') as f:
    data = json.load(f)
models = {}
for exp in data['experiments']:
    m = exp['model']
    if m not in models:
        models[m] = {'completed': 0, 'errors': 0, 'error_msgs': set()}
    if exp['status'] == 'COMPLETED':
        models[m]['completed'] += 1
    else:
        models[m]['errors'] += 1
        models[m]['error_msgs'].add(exp.get('error', 'unknown'))
for m, v in sorted(models.items()):
    print(f'{m}: completed={v["completed"]}, errors={v["errors"]}')
    for err in v['error_msgs']:
        print(f'  ERROR: {err}')
print('Total completed:', sum(v['completed'] for v in models.values()))
print('Total errors:', sum(v['errors'] for v in models.values()))