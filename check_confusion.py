import json
import numpy as np

with open('training/models/rf_emotion_200est_15depth_metadata.json', 'r') as f:
    meta = json.load(f)

import numpy as np
cm = np.array(meta['metrics']['test']['confusion_matrix'])
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
print('Confusion Matrix:')
print('      ', '  '.join(f'{l[:4]:>4}' for l in labels))
for i, row in enumerate(cm):
    print(f'{labels[i][:4]:>4}', ' '.join(f'{v:>4}' for v in row))
print()
for label, metrics in meta['metrics']['test']['per_class'].items():
    print(f"{label}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} support={metrics['support']}")