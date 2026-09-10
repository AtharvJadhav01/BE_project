import os, json, numpy as np, sys

sys.path.insert(0, 'training')
sys.path.insert(0, 'src')

data_dir = r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work'

# Check fer2013_features_v2
v2_dir = os.path.join(data_dir, 'training', 'fer2013_features_v2')
print('v2_dir exists:', os.path.exists(v2_dir))

if os.path.exists(v2_dir):
    meta_path = os.path.join(v2_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)
    print(f'feature_dim: {meta["feature_dim"]}')
    print(f'feature names count: {len(meta["feature_names"])}')
    
    X_train = np.load(os.path.join(v2_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(v2_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(v2_dir, 'X_test.npy'))
    print(f'X_train shape: {X_train.shape}')
    print(f'X_val shape: {X_val.shape}')
    print(f'X_test shape: {X_test.shape}')

# Also check if we can load from the original location concept
print()
print('Looking for any dataset artifacts...')
for root, dirs, files in os.walk(data_dir):
    if 'metadata.json' in files:
        print(f'Found metadata at: {root}')
        meta_path = os.path.join(root, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        print(f'  feature_dim: {meta["feature_dim"]}')
        print(f'  feature names count: {len(meta["feature_names"])}')
        break