import json

# Check v2 metadata
with open('training/fer2013_features_v2/metadata.json') as f:
    v2 = json.load(f)
print('=== V2 METADATA ===')
print('feature_dim:', v2['feature_dim'])
print('num feature names:', len(v2['feature_names']))
# Check for duplicates
names = v2['feature_names']
dups = [name for name in set(names) if names.count(name) > 1]
print('Duplicate names:', dups)
print('Unique names:', len(set(names)))

# Check expanded metadata
with open('training/fer2013_features_expanded/metadata.json') as f:
    exp = json.load(f)
print()
print('=== EXPANDED METADATA ===')
print('feature_dim:', exp['feature_dim'])
print('num feature names:', len(exp['feature_names']))
names2 = exp['feature_names']
# Count by category prefix
lm_count = sum(1 for n in names2 if n.startswith('lm_'))
bs_count = sum(1 for n in names2 if n.startswith('bs_'))
head_count = sum(1 for n in names2 if n.startswith('head_'))
other = len(names2) - lm_count - bs_count - head_count
print('Landmark (lm_):', lm_count)
print('Blendshape (bs_):', bs_count)
print('Head pose (head_):', head_count)
print('Other:', other)