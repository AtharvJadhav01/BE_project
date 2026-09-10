import sys

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/training/feature_extractor.py', 'r') as f:
    content = f.read()

# Find and replace the get_feature_names method
old = '''    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            # Eye features (original + new)
            "eye_openness_left", "eye_openness_right", "eye_openness_avg",
            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate",
            "eye_openness_asymmetry", "eye_squint_left", "eye_squint_right",
            "eye_wide_left", "eye_wide_right",
            
            # Mouth features
            "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry",
            # Additional mouth features for emotion estimation
            "mouth_funnel", "mouth_pucker", "mouth_stretch_left", "mouth_stretch_right",
            
            # Eyebrow features
            "brow_inner_up", "brow_down_left", "brow_down_right",
            "brow_outer_up_left", "brow_outer_up_right",
            
            # Head pose
            "head_pitch", "head_yaw", "head_roll",
            
            # Facial movement
            "facial_movement_velocity", "facial_symmetry",
            
            # Overall
            "overall_expressivity",
            
            # Nose features for emotion estimation (disgust, etc.)
            "nose_sneer_left", "nose_sneer_right"
        ]'''

new = '''    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            # Eye features
            "eye_openness_left", "eye_openness_right", "eye_openness_avg",
            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate",
            "eye_openness_asymmetry", "eye_squint_left", "eye_squint_right",
            "eye_wide_left", "eye_wide_right",
            
            # Mouth features
            "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry",
            "mouth_funnel", "mouth_pucker", "mouth_stretch_left", "mouth_stretch_right",
            "mouth_aspect_ratio", "mouth_height", "upper_lip_raise", "lower_lip_depressor",
            "chin_raise", "mouth_corner_displacement",
            
            # Eyebrow features
            "brow_inner_up", "brow_down_left", "brow_down_right",
            "brow_outer_up_left", "brow_outer_up_right",
            "brow_raise_ratio", "brow_furrow_intensity",
            "brow_eye_distance_left", "brow_eye_distance_right",
            
            # Head pose
            "head_pitch", "head_yaw", "head_roll",
            
            # Facial movement
            "facial_movement_velocity", "facial_symmetry",
            
            # Overall expressivity
            "overall_expressivity", "upper_face_expressivity", "lower_face_expressivity",
            "expressivity_ratio",
            
            # Nose features
            "nose_sneer_left", "nose_sneer_right", "nose_wrinkle_intensity",
        ]'''

if 'def get_feature_names(self) -> List[str]:' in content:
    content = content.replace(old, new)
    with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/training/feature_extractor.py', 'w') as f:
        f.write(content)
    print('Fixed feature names!')
else:
    print('Pattern not found')