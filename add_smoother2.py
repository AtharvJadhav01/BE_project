#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')

# Read the file
with open('src/face/capture.py', 'r') as f:
    content = f.read()

# Now update the _poll_face_analysis method in main_window.py to use the smoother
# First, let's find where expression_estimate is set and add smoother integration

# We need to add smoother.add_prediction() call when we have an expression result
# And use get_smoothed() to get the smoothed expression

# Let's find the section where expression_estimate is created/used
idx = content.find("expression_estimate = None")
if idx != -1:
    print(f"Found expression_estimate = None at index {idx}")
    # Print surrounding context
    print(content[max(0, idx-50):idx+150])
else:
    print("Not found - searching for expression estimate pattern")
    # Try alternative
    idx = content.find("expression_estimate")
    if idx != -1:
        print(f"Found expression_estimate at index {idx}")
        print(content[max(0, idx-50):idx+150])