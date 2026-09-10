import sys

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/training/feature_extractor.py', 'r') as f:
    content = f.read()

# Remove the old get_feature_names from FacialFeatures class
idx_start = content.find('def get_feature_names')
idx_end = content.find('    def ', idx_start + 1)
if idx_end < 0:
    idx_end = content.find('class ', idx_start + 1)
if idx_end < 0:
    idx_end = len(content)

# Remove the old method
content = content[:idx_start] + content[idx_end:]

# Now find the MediaPipeFaceFeatureExtractor class close method
idx_close = content.find('def close(self):')
if idx_close >= 0:
    # Insert the new method before close
    new_method = '''    def get_feature_names(self) -> List[str]:
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
        ]

'''

    if 'def close(self):' in content:
        # Find the close method in MediaPipeFaceFeatureExtractor class
        idx_mediapipe = content.find('class MediaPipeFaceFeatureExtractor:')
        idx_close = content.find('def close(self):', content.find('class MediaPipeFaceFeatureExtractor:'))
        
        if idx_close >= 0:
            new_content = content[:idx_close] + '\n    def get_feature_names(self) -> List[str]:\n        """Return list of feature names this extractor produces."""\n        return [\n            # Eye features\n            "eye_openness_left", "eye_openness_right", "eye_openness_avg",\n            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate",\n            "eye_openness_asymmetry", "eye_squint_left", "eye_squint_right",\n            "eye_wide_left", "eye_wide_right",\n            \n            # Mouth features\n            "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry",\n            "mouth_funnel", "mouth_pucker", "mouth_stretch_left", "mouth_stretch_right",\n            "mouth_aspect_ratio", "mouth_height", "upper_lip_raise", "lower_lip_depressor",\n            "chin_raise", "mouth_corner_displacement",\n            \n            # Eyebrow features\n            "brow_inner_up", "brow_down_left", "brow_down_right",\n            "brow_outer_up_left", "brow_outer_up_right",\n            "brow_raise_ratio", "brow_furrow_intensity",\n            "brow_eye_distance_left", "brow_eye_distance_right",\n            \n            # Head pose\n            "head_pitch", "head_yaw", "head_roll",\n            \n            # Facial movement\n            "facial_movement_velocity", "facial_symmetry",\n            \n            # Overall expressivity\n            "overall_expressivity", "upper_face_expressivity", "lower_face_expressivity",\n            "expressivity_ratio",\n            \n            # Nose features\n            "nose_sneer_left", "nose_sneer_right", "nose_wrinkle_intensity",\n        ]\n\n' + content[idx_close:]
            with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/training/feature_extractor.py', 'w') as f:
                f.write(content)
            print('Moved successfully!')
        else:
            print('close method not found')
    else:
        print('Already moved or marker not found')