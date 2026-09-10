import sys

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/training/feature_extractor.py', 'r') as f:
    content = f.read()

# Check if methods already exist
if '_calculate_expressivity_components' in content:
    print('Methods already exist!')
    sys.exit(0)

# Find the close method and insert before it
old = '''        except Exception:
            pass

    def close(self):'''

new = '''        except Exception:
            pass

    def _calculate_expressivity_components(self, features: FacialFeatures) -> None:
        """Calculate upper vs lower face expressivity components."""
        try:
            # Upper face: eyes + eyebrows
            eye_expr = (features.eye_openness_left + features.eye_openness_right) / 2.0
            brow_expr = (features.brow_inner_up + 
                        (features.brow_down_left + features.brow_down_right) / 2.0 +
                        features.brow_outer_up_left + 
                        features.brow_outer_up_right) / 4.0
            features.upper_face_expressivity = (eye_expr + brow_expr) / 2.0
            
            # Lower face: mouth
            mouth_expr = features.mouth_width
            features.lower_face_expressivity = mouth_expr
            
            # Ratio
            if features.lower_face_expressivity > 0.01:
                features.expressivity_ratio = features.upper_face_expressivity / features.lower_face_expressivity
            else:
                features.expressivity_ratio = features.upper_face_expressivity * 10.0
        except Exception:
            features.upper_face_expressivity = 0.0
            features.lower_face_expressivity = 0.0
            features.expressivity_ratio = 0.0

    def _calculate_geometric_features(self, landmarks_3d: List[Dict[str, float]], features: FacialFeatures) -> None:
        """Calculate geometric features from landmarks."""
        try:
            if len(landmarks_3d) < 468:
                return
            
            # Key landmark indices
            MOUTH_LEFT = 61
            MOUTH_RIGHT = 291
            LEFT_EYE_INNER = 133
            RIGHT_EYE_INNER = 362
            LEFT_BROW_INNER = 70
            RIGHT_BROW_INNER = 300
            
            # Face center for normalization
            face_width = self._get_face_width(landmarks_3d)
            if face_width < 0.01:
                face_width = 0.1
            
            # 1. Mouth Aspect Ratio (MAR) - mouth height / width
            if len(landmarks_3d) > 14 and len(landmarks_3d) > 61:
                mouth_height = abs(landmarks_3d[13]["y"] - landmarks_3d[14]["y"])
                mouth_width = abs(landmarks_3d[291]["x"] - landmarks_3d[61]["x"])
                if mouth_width > 0:
                    features.mouth_aspect_ratio = mouth_height / mouth_width
                features.mouth_height = mouth_height
            
            # 2. Mouth corner displacement from neutral
            if len(landmarks_3d) > 291 and len(landmarks_3d) > 61:
                mouth_center_x = (landmarks_3d[61]["x"] + landmarks_3d[291]["x"]) / 2
                mouth_center_y = (landmarks_3d[61]["y"] + landmarks_3d[291]["y"]) / 2
                corner_disp = math.sqrt(
                    (landmarks_3d[61]["x"] - mouth_center_x)**2 + 
                    (landmarks_3d[61]["y"] - mouth_center_y)**2
                )
                features.mouth_corner_displacement = corner_disp / face_width
            
            # 3. Eye-brow distances
            if len(landmarks_3d) > 70 and len(landmarks_3d) > 133:
                left_brow_y = landmarks_3d[70]["y"]
                left_eye_y = landmarks_3d[133]["y"]
                features.eye_brow_distance_left = abs(left_brow_y - left_eye_y) / face_width
                
            if len(landmarks_3d) > 300 and len(landmarks_3d) > 362:
                right_brow_y = landmarks_3d[300]["y"]
                right_eye_y = landmarks_3d[362]["y"]
                features.eye_brow_distance_right = abs(right_brow_y - right_eye_y) / face_width
            
            # 4. Brow raise ratio (inner up vs down)
            brow_up = features.brow_inner_up
            brow_down = (features.brow_down_left + features.brow_down_right) / 2.0
            if brow_down > 0.01:
                features.brow_raise_ratio = brow_up / brow_down
            else:
                features.brow_raise_ratio = brow_up * 10.0
            
            # 5. Brow furrow intensity
            features.brow_furrow_intensity = (features.brow_down_left + features.brow_down_right) / 2.0
            
            # 6. Eye asymmetry
            features.eye_openness_asymmetry = abs(features.eye_openness_left - features.eye_openness_right)
            
            # 7. Nose wrinkle intensity
            features.nose_wrinkle_intensity = (features.nose_sneer_left + features.nose_sneer_right) / 2.0
            
        except Exception:
            pass

    def close(self):