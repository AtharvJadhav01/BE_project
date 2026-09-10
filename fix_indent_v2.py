with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/analysis/face/features.py', 'r') as f:
    content = f.read()

# Find the _extract_eyebrow_features method
idx_start = content.find('def _extract_eyebrow_features')
if idx_start < 0:
    print('Method not found!')
    exit(1)

# Find the end of the method (next method definition)
idx_end = content.find('def _extract_facial_dynamics', idx_start)
if idx_end < 0:
    idx_end = content.find('def _calculate_head_pose', idx_start)
if idx_end < 0:
    idx_end = content.find('def ', idx_start + 1)

if idx_end < 0:
    idx_end = len(content)

# Extract the method
method_text = content[idx_start:idx_end]

# Fix indentation - add 4 spaces to each line
lines = method_text.split('\n')
fixed_lines = []
for line in lines:
    if line.strip() == '':
        fixed_lines.append('')
    else:
        fixed_lines.append('    ' + line.lstrip())

fixed_method = '\n'.join(fixed_lines)

# Replace in content
content = content[:idx_start] + fixed_method + content[idx_end:]

# Also fix the _extract_mouth_features method if it has the same issue
idx_mouth = content.find('def _extract_mouth_features')
if idx_mouth >= 0:
    idx_end_mouth = content.find('def _', idx_mouth + 1)
    if idx_end_mouth < 0:
        idx_end_mouth = len(content)
    method_text = content[idx_mouth:idx_end_mouth]
    lines = method_text.split('\n')
    fixed_lines = []
    for line in lines:
        if line.strip() == '':
            fixed_lines.append('')
        else:
            fixed_lines.append('    ' + line.lstrip())
    fixed_method = '\n'.join(fixed_lines)
    # Find the exact start and end
    idx_start_mouth = content.find('def _extract_mouth_features')
    idx_end_mouth = content.find('def _', idx_mouth + 1)
    if idx_end_mouth < 0:
        idx_end_mouth = len(content)
    content = content[:idx_start_mouth] + '\n'.join(fixed_lines) + content[idx_end_mouth:]

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/analysis/face/features.py', 'w') as f:
    f.write(content)
print('Fixed indentation!')