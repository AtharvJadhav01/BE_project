#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')

# Read the file
with open('src/face/capture.py', 'r') as f:
    content = f.read()

# Add EmotionSmoother initialization after emotion_enabled check
old_code = """# Check if emotion estimation is enabled
            self._emotion_enabled = face_config.get("emotion_enabled", True)"""

new_code = """# Check if emotion estimation is enabled
            self._emotion_enabled = face_config.get("emotion_enabled", True)
            
            # Initialize emotion smoother
            smoothing_window = face_config.get("emotion_smoothing_window", 5)
            smoothing_alpha = face_config.get("emotion_smoothing_alpha", 0.3)
            confidence_threshold = face_config.get("emotion_confidence_threshold", 0.5)
            self._emotion_smoother = EmotionSmoother(
                window_size=smoothing_window,
                alpha=smoothing_alpha,
                min_confidence=confidence_threshold,
                enabled=self._emotion_enabled
            )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('src/face/capture.py', 'w') as f:
        f.write(content)
    print('EmotionSmoother init added successfully')
else:
    print('Pattern not found')
    # Try to find the line
    idx = content.find('# Check if emotion estimation is enabled')
    if idx != -1:
        print(f'Found at index {idx}')
        print(content[idx:idx+100])