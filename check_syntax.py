import sys

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/training/feature_extractor.py', 'r') as f:
    content = f.read()

# Check if _calculate_expressivity exists (the original method)
if '_calculate_expressivity_components' not in content:
    print('Methods not found, checking for _calculate_expressivity...')
    if '_calculate_expressivity' in content:
        print('Found _calculate_expressivity - need to check if _calculate_expressivity_components exists')
    
# Check if methods exist
print('_calculate_expressivity_components' in content)
print('_calculate_geometric_features' in content)

# Check the exact structure around the close method
idx = content.find('def close(self):')
if idx > 0:
    print('Found close method at:', idx)
    print(repr(content[idx-100:idx+100]))
else:
    print('close method not found')