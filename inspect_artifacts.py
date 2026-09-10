import os, json, numpy as np

work_dir = r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work'
os.chdir(work_dir)
training_dir = os.path.join(work_dir, 'training')
v2_dir = os.path.join(training_dir, 'fer2013_features_v2')

# Check metadata
with open(os.path.join(v2_dir, 'metadata.json')) as f:
    meta = json.load(f)
print('v2 Feature dim:', meta['feature_dim'])
print('v2 Feature names count:', len(meta['feature_names']))
print('v2 Feature names (first 10):', meta['feature_names'][:10])

# Check X shapes
X_train = np.load(os.path.join(v2_dir, 'X_train.npy'))
X_val = np.load(os.path.join(v2_dir, 'X_val.npy'))
X_test = np.load(os.path.join(v2_dir, 'X_test.npy'))
print('v2 X_train shape:', X_train.shape)
print('v2 X_val shape:', X_val.shape)
print('v2 X_test shape:', X_test.shape)

# Also check the original fer2013_features directory
orig_dir = os.path.join(training_dir, 'fer2013_features')
print()
print('Original dir exists:', os.path.exists(orig_dir))
if os.path.exists(orig_dir):
    for f in os.listdir(orig_dir):
        print(' ', f)