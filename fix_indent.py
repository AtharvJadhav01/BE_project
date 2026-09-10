import sys

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/analysis/face/features.py', 'r') as f:
    content = f.read()

# Fix the indentation error in _extract_eyebrow_features
old = '''    def _extract_eyebrow_features(self, blendshapes: Dict[str, float], 
                                 features: FacialFeatures) -> None:
        try:
            features.brow_inner_up = blendshapes.get("browInnerUp", 0.0)
            features.brow_down_left = blendshapes.get("browDownLeft", 0.0)
            features.brow_down_right = blendshapes.get("browDownRight", 0.0)
            features.brow_outer_up_left = blendshapes.get("browOuterUpLeft", 0.0)
            features.brow_outer_up_right = blendshapes.get("browOuterUpRight", 0.0)
        except Exception:
            features.brow_inner_up = 0.0
            features.brow_down_left = 0.0
            features.brow_down_right = 0.0
            features.brow_outer_up_left = 0.0
            features.brow_outer_up_right = 0.0'''

new = '''    def _extract_eyebrow_features(self, blendshapes: Dict[str, float], 
                                 features: FacialFeatures) -> None:
        """Extract eyebrow-related features."""
        try:
            features.brow_inner_up = blendshapes.get("browInnerUp", 0.0)
            features.brow_down_left = blendshapes.get("browDownLeft", 0.0)
            features.brow_down_right = blendshapes.get("browDownRight", 0.0)
            features.brow_outer_up_left = blendshapes.get("browOuterUpLeft", 0.0)
            features.brow_outer_up_right = blendshapes.get("browOuterUpRight", 0.0)
        except Exception:
            features.brow_inner_up = 0.0
            features.brow_down_left = 0.0
            features.brow_down_right = 0.0
            features.brow_outer_up_left = 0.0
            features.brow_outer_up_right = 0.0'''

if old in content:
    content = content.replace(old, new)
    with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/analysis/face/features.py', 'w') as f:
        f.write(content)
    print('Fixed!')
else:
    print('Pattern not found')