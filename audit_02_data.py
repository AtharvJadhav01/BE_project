import numpy as np
import json
import os

# Check actual feature dimensions in the dataset
v2_dir = 'training/fer2013_features_v2'
exp_dir = 'training/fer2013_features_expanded'

# List files in v2 dir
print("V2 directory files:")
for f in os.listdir(v2_dir):
    if f.endswith('.npy') or f.endswith('.json'):
        print("  " + f)

# List files in expanded dir
print("\nExpanded directory files:")
for f in os.listdir(exp_dir):
    if f.endswith('.npy') or f.endswith('.json'):
        print("  " + f)

# Try to load a sample from v2
data_path = os.path.join(v2_dir, 'data.npy')
if os.path.exists(data_path):
    data = np.load(data_path, allow_pickle=True)
    print("\nV2 data shape:", data.shape)
    if len(data) > 0:
        first = data[0]
        print("First sample features length:", len(first) if hasattr(first, '__len__') else 'scalar')
        if hasattr(first, '__len__') and len(first) > 0:
            print("First sample first 10 features:", first[:10])
else:
    print("\nV2 data.npy not found")

# Try to load a sample from expanded
data_path2 = os.path.join(exp_dir, 'data.npy')
if os.path.exists(data_path2):
    data2 = np.load(data_path2, allow_pickle=True)
    print("\nExpanded data shape:", data2.shape)
    if len(data2) > 0:
        first2 = data2[0]
        print("First sample feature count:", len(first2))
        if hasattr(first2, '__len__') and len(first2) > 0:
            print("First sample first 10 features:", first2[:10])
else:
    print("\nExpanded data.npy not found")