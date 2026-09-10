"""
Facial Expression Training Pipeline - Feature Extraction Module

This module reuses the application's MediaPipe feature extraction logic
to extract 27-dimensional facial features from static images.
"""

import os
import sys
import math
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# MediaPipe imports
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat


@dataclass
class FacialFeatures:
    """Container for extracted facial features."""
    # Eye features
    eye_openness_left: float = 0.0
    eye_openness_right: float = 0.0
    eye_openness_avg: float = 0.0
    eye_aspect_ratio_left: float = 0.0
    eye_aspect_ratio_right: float = 0.0
    blink_rate: float = 0.0
    
    # Mouth features
    mouth_openness: float = 0.0
    mouth_width: float = 0.0
    lip_press: float = 0.0
    smile_asymmetry: float = 0.0
    # Phase 1 additions
    mouth_funnel: float = 0.0
    mouth_pucker: float = 0.0
    mouth_stretch_left: float = 0.0
    mouth_stretch_right: float = 0.0
    
    # Eyebrow features
    brow_inner_up: float = 0.0
    brow_down_left: float = 0.0
    brow_down_right: float = 0.0
    brow_outer_up_left: float = 0.0
    brow_outer_up_right: float = 0.0
    
    # Head pose
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    
    # Facial movement and dynamics
    facial_movement_velocity: float = 0.0
    facial_symmetry: float = 1.0
    
    # Overall expressivity
    overall_expressivity: float = 0.0
    
    # Nose features for emotion estimation
    nose_sneer_left: float = 0.0
    nose_sneer_right: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        return np.array(list(self.to_dict().values()), dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        return [
            "eye_openness_left", "eye_openness_right", "eye_openness_avg",
            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate",
            "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry",
            "mouth_funnel", "mouth_pucker", "mouth_stretch_left", "mouth_stretch_right",
            "brow_inner_up", "brow_down_left", "brow_down_right",
            "brow_outer_up_left", "brow_outer_up_right",
            "head_pitch", "head_yaw", "head_roll",
            "facial_movement_velocity", "facial_symmetry",
            "overall_expressivity",
            "nose_sneer_left", "nose_sneer_right"
        ]


class MediaPipeFaceFeatureExtractor:
    """
    Standalone MediaPipe-based feature extractor for static images.
    
    Reuses the same feature extraction logic as the live application.
    """
    
    # MediaPipe blendshape indices (matching application)
    BLENDSHAPE_INDICES = {
        "eyeBlinkLeft": 4, "eyeBlinkRight": 5, "eyeLookDownLeft": 6,
        "eyeLookDownRight": 7, "eyeLookInLeft": 8, "eyeLookInRight": 9,
        "eyeLookOutLeft": 10, "eyeLookOutRight": 11, "eyeLookUpLeft": 12,
        "eyeLookUpRight": 13, "eyeSquintLeft": 14, "eyeSquintRight": 15,
        "eyeWideLeft": 16, "eyeWideRight": 17,
        
        "browDownLeft": 0, "browDownRight": 1, "browInnerUp": 2,
        "browOuterUpLeft": 3, "browOuterUpRight": 4,
        
        "mouthClose": 28, "mouthFrownLeft": 30, "mouthFrownRight": 31,
        "mouthFunnel": 32, "mouthLeft": 33, "mouthPucker": 37,
        "mouthRight": 38, "mouthShrugLower": 39, "mouthShrugUpper": 40,
        "mouthSmileLeft": 41, "mouthSmileRight": 42,
        "mouthStretchLeft": 43, "mouthStretchRight": 44,
        "mouthUpperUpLeft": 45, "mouthUpperUpRight": 46,
        "mouthDimpleLeft": 29, "mouthDimpleRight": 34,
        "mouthPressLeft": 35, "mouthPressRight": 36,
        "mouthRollLower": 47, "mouthRollUpper": 48,
        "mouthShrugLower": 39, "mouthShrugUpper": 40,
        "mouthDimpleLeft": 29, "mouthDimpleRight": 34,
        
        "jawOpen": 18, "jawForward": 19, "jawLeft": 20, "jawRight": 21,
        
        "noseSneerLeft": 50, "noseSneerRight": 51
    }
    
    # Key landmark indices for geometric calculations
    LANDMARK_INDICES = {
        "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 162, 246],
        "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        
        "mouth": [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        
        "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
        
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_ear": 234,
        "right_ear": 454
    }
    
    def __init__(self, model_path: str):
        """Initialize MediaPipe FaceLandmarker."""
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[FacialFeatures], bool]:
        """
        Extract facial features from a single image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Tuple of (FacialFeatures or None, face_detected_flag)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create MediaPipe Image
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
        
        # Process image
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None, False
        
        face_landmarks = result.face_landmarks[0]
        
        # Get detection confidence
        detection_confidence = 1.0
        try:
            visibilities = [lm.visibility for lm in face_landmarks if hasattr(lm, 'visibility')]
            if visibilities:
                detection_confidence = float(np.mean(visibilities))
        except Exception:
            detection_confidence = 1.0
        
        # Extract landmarks
        landmarks_3d = []
        landmarks_2d = []
        for lm in face_landmarks:
            landmarks_3d.append({"x": lm.x, "y": lm.y, "z": lm.z})
            landmarks_2d.append({"x": int(lm.x * w), "y": int(lm.y * h)})
        
        # Calculate bounding box
        xs = [lm["x"] for lm in landmarks_2d]
        ys = [lm["y"] for lm in landmarks_2d]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bounding_box = {
            "x": x_min, "y": y_min,
            "width": x_max - x_min, "height": y_max - y_min
        }
        
        # Extract blendshapes
        blendshapes = {}
        if result.face_blendshapes:
            for i, blendshape in enumerate(result.face_blendshapes[0]):
                if i < len(self._get_blendshape_names()):
                    blendshapes[self._get_blendshape_names()[i]] = float(blendshape.score)
        
        # Extract iris
        iris = None
        try:
            if len(face_landmarks) > 473:
                left_iris = face_landmarks[468]
                right_iris = face_landmarks[473]
                iris = {
                    "left": {"x": left_iris.x, "y": left_iris.y, "z": left_iris.z},
                    "right": {"x": right_iris.x, "y": right_iris.y, "z": right_iris.z}
                }
        except Exception:
            pass
        
        # Calculate head pose
        head_pose = self._calculate_head_pose(face_landmarks, w, h)
        
        # Build raw data dict (matching application format)
        raw_data = {
            "face_detected": True,
            "landmarks_3d": landmarks_3d,
            "landmarks_2d": landmarks_2d,
            "blendshapes": blendshapes,
            "iris": iris,
            "head_pose": head_pose
        }
        
        # Extract features using application logic
        features = self._extract_features_from_raw(raw_data)
        
        return features, True
    
    def _get_blendshape_names(self) -> List[str]:
        """Get ordered blendshape names matching MediaPipe output."""
        return [
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
            "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
            "jawLeft", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
            "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
            "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
            "noseSneerLeft", "noseSneerRight"
        ]
    
    def _extract_features_from_raw(self, raw_data: Dict[str, Any]) -> FacialFeatures:
        """Extract features using the same logic as the application."""
        features = FacialFeatures()
        
        landmarks_3d = raw_data.get("landmarks_3d", [])
        blendshapes = raw_data.get("blendshapes", {})
        head_pose = raw_data.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
        
        if not landmarks_3d:
            return features
        
        # Extract eye features
        self._extract_eye_features(landmarks_3d, blendshapes, features)
        
        # Extract mouth features
        self._extract_mouth_features(landmarks_3d, blendshapes, features)
        
        # Extract eyebrow features
        self._extract_eyebrow_features(blendshapes, features)
        
        # Extract nose features
        self._extract_nose_features(blendshapes, features)
        
        # Extract head pose
        features.head_pitch = head_pose.get("pitch", 0.0)
        features.head_yaw = head_pose.get("yaw", 0.0)
        features.head_roll = head_pose.get("roll", 0.0)
        
        # For static images, movement and symmetry are computed but blink_rate is 0
        features.facial_movement_velocity = 0.0
        features.facial_symmetry = self._calculate_facial_symmetry(raw_data.get("landmarks_3d", []))
        
        # Calculate overall expressivity
        features.overall_expressivity = self._calculate_expressivity(features)
        
        # blink_rate is 0 for static images
        features.blink_rate = 0.0
        
        return features
    
    def _get_blendshape_names(self) -> List[str]:
        """Get ordered blendshape names matching MediaPipe output."""
        return [
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
            "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
            "jawLeft", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
            "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
            "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
            "noseSneerLeft", "noseSneerRight"
        ]
    
    def _extract_eye_features(self, landmarks_3d: List[Dict[str, float]], 
                             blendshapes: Dict[str, float], 
                             features: FacialFeatures) -> None:
        try:
            left_ear = self._calculate_eye_aspect_ratio(landmarks_3d, "left")
            right_ear = self._calculate_eye_aspect_ratio(landmarks_3d, "right")
            
            features.eye_aspect_ratio_left = left_ear
            features.eye_aspect_ratio_right = right_ear
            
            features.eye_openness_left = max(0.0, min(1.0, (left_ear - 0.2) / 0.3))
            features.eye_openness_right = max(0.0, min(1.0, (right_ear - 0.2) / 0.3))
            features.eye_openness_avg = (features.eye_openness_left + features.eye_openness_right) / 2.0
            
            # Blink detection from blendshapes
            left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
            right_blink = blendshapes.get("eyeBlinkRight", 0.0)
            blink_value = max(left_blink, right_blink)
            
            # Blink rate is 0 for static images (would need temporal tracking)
            features.blink_rate = 0.0
        except Exception:
            features.eye_openness_left = 0.0
            features.eye_openness_right = 0.0
            features.eye_openness_avg = 0.0
            features.eye_aspect_ratio_left = 0.0
            features.eye_aspect_ratio_right = 0.0
            features.blink_rate = 0.0
    
    def _calculate_eye_aspect_ratio(self, landmarks_3d: List[Dict[str, float]], eye: str) -> float:
        if eye == "left":
            points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 162]
        else:
            points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        if len(landmarks_3d) < max(points) + 1:
            return 0.3
        
        try:
            p1 = landmarks_3d[points[0]]
            p2 = landmarks_3d[points[1]]
            p3 = landmarks_3d[points[2]]
            p4 = landmarks_3d[points[3]]
            p5 = landmarks_3d[points[4]]
            p6 = landmarks_3d[points[5]]
            
            vert1 = self._euclidean_distance(p2, p6)
            vert2 = self._euclidean_distance(p3, p5)
            horiz = self._euclidean_distance(p1, p4)
            
            if horiz == 0:
                return 0.0
                
            ear = (vert1 + vert2) / (2.0 * horiz)
            return ear
        except Exception:
            return 0.3
    
    def _euclidean_distance(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        return math.sqrt(
            (p1["x"] - p2["x"])**2 + 
            (p1["y"] - p2["y"])**2 + 
            (p1["z"] - p2["z"])**2
        )
    
    def _extract_mouth_features(self, landmarks_3d: List[Dict[str, float]], 
                               blendshapes: Dict[str, float], 
                               features: FacialFeatures) -> None:
        try:
            features.mouth_openness = blendshapes.get("jawOpen", 0.0)
            
            smile_left = blendshapes.get("mouthSmileLeft", 0.0)
            smile_right = blendshapes.get("mouthSmileRight", 0.0)
            features.mouth_width = (smile_left + smile_right) / 2.0
            
            features.lip_press = blendshapes.get("mouthPressLeft", 0.0) + blendshapes.get("mouthPressRight", 0.0)
            features.smile_asymmetry = abs(smile_left - smile_right)
            
            # Phase 1 additions
            features.mouth_funnel = blendshapes.get("mouthFunnel", 0.0)
            features.mouth_pucker = blendshapes.get("mouthPucker", 0.0)
            features.mouth_stretch_left = blendshapes.get("mouthStretchLeft", 0.0)
            features.mouth_stretch_right = blendshapes.get("mouthStretchRight", 0.0)
        except Exception:
            features.mouth_openness = 0.0
            features.mouth_width = 0.0
            features.lip_press = 0.0
            features.smile_asymmetry = 0.0
            features.mouth_funnel = 0.0
            features.mouth_pucker = 0.0
            features.mouth_stretch_left = 0.0
            features.mouth_stretch_right = 0.0
    
    def _extract_eyebrow_features(self, blendshapes: Dict[str, float], 
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
            features.brow_outer_up_right = 0.0
    
    def _extract_nose_features(self, blendshapes: Dict[str, float], 
                                features: FacialFeatures) -> None:
        try:
            features.nose_sneer_left = blendshapes.get("noseSneerLeft", 0.0)
            features.nose_sneer_right = blendshapes.get("noseSneerRight", 0.0)
        except Exception:
            features.nose_sneer_left = 0.0
            features.nose_sneer_right = 0.0
    
    def _calculate_head_pose(self, face_landmarks, w: int, h: int) -> Dict[str, float]:
        try:
            NOSE_TIP = 1
            LEFT_EYE_OUTER = 33
            RIGHT_EYE_OUTER = 263
            CHIN = 152
            
            nose_tip = face_landmarks[NOSE_TIP]
            left_eye = face_landmarks[LEFT_EYE_OUTER]
            right_eye = face_landmarks[RIGHT_EYE_OUTER]
            chin = face_landmarks[CHIN]
            
            nose_2d = np.array([nose_tip.x * w, nose_tip.y * h])
            left_eye_2d = np.array([left_eye.x * w, left_eye.y * h])
            right_eye_2d = np.array([right_eye.x * w, right_eye.y * h])
            chin_2d = np.array([chin.x * w, chin.y * h])
            
            eye_center = (left_eye_2d + right_eye_2d) / 2
            nose_to_eye_center = nose_2d - eye_center
            eye_vector = right_eye_2d - left_eye_2d
            eye_distance = np.linalg.norm(eye_vector)
            
            if eye_distance > 0:
                yaw = np.degrees(np.arctan2(nose_to_eye_center[0], eye_distance * 0.5))
                
                eye_to_chin = chin_2d - eye_center
                nose_to_eye = nose_2d - eye_center
                pitch = np.degrees(np.arctan2(nose_to_eye[1], np.linalg.norm(eye_to_chin) * 0.5))
                
                roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            else:
                yaw = pitch = roll = 0.0
            
            return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
        except Exception:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    
    def _calculate_facial_symmetry(self, landmarks_3d: List[Dict[str, float]]) -> float:
        try:
            symmetrical_pairs = [
                (33, 263), (133, 362), (70, 300), (105, 334),
                (61, 291), (78, 308), (1, 1), (199, 199)
            ]
            
            total_asymmetry = 0.0
            valid_pairs = 0
            
            face_width = self._get_face_width(landmarks_3d)
            
            if face_width > 0:
                for left_idx, right_idx in symmetrical_pairs:
                    if left_idx < len(landmarks_3d) and right_idx < len(landmarks_3d):
                        left_point = landmarks_3d[left_idx]
                        right_point = landmarks_3d[right_idx]
                        
                        midline_x = 0.5
                        left_distance = abs(left_point["x"] - midline_x)
                        right_distance = abs(right_point["x"] - midline_x)
                        
                        asymmetry = abs(left_distance - right_distance)
                        normalized_asymmetry = asymmetry / face_width
                        
                        total_asymmetry += normalized_asymmetry
                        valid_pairs += 1
                
                if valid_pairs > 0:
                    avg_asymmetry = total_asymmetry / valid_pairs
                    symmetry = max(0.0, 1.0 - (avg_asymmetry * 5.0))
                    return min(1.0, max(0.0, symmetry))
            
            return 0.5
        except Exception:
            return 0.5
    
    def _get_face_width(self, landmarks_3d: List[Dict[str, float]]) -> float:
        try:
            if len(landmarks_3d) > 454:
                left_ear = landmarks_3d[234]
                right_ear = landmarks_3d[454]
                dx = right_ear["x"] - left_ear["x"]
                dy = right_ear["y"] - left_ear["y"]
                dz = right_ear["z"] - left_ear["z"]
                return math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if len(landmarks_3d) > 362:
                left_eye = landmarks_3d[33]
                right_eye = landmarks_3d[362]
                dx = right_eye["x"] - left_eye["x"]
                dy = right_eye["y"] - left_eye["y"]
                dz = right_eye["z"] - left_eye["z"]
                return math.sqrt(dx*dx + dy*dy + dz*dz)
            
            return 0.1
        except Exception:
            return 0.1
    
    def _calculate_expressivity(self, features: FacialFeatures) -> float:
        try:
            eye_expr = (features.eye_openness_left + features.eye_openness_right) / 2.0
            mouth_expr = features.mouth_width
            brow_expr = (features.brow_inner_up + 
                        (features.brow_down_left + features.brow_down_right) / 2.0 +
                        features.brow_outer_up_left + 
                        features.brow_outer_up_right) / 4.0
            movement_expr = min(1.0, features.facial_movement_velocity * 10.0)
            
            expressivity = (
                eye_expr * 0.3 +
                mouth_expr * 0.3 +
                brow_expr * 0.2 +
                movement_expr * 0.2
            )
            return min(1.0, max(0.0, expressivity))
        except Exception:
            return 0.0
    
    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


def create_extractor(model_path: str = "face_landmarker.task") -> MediaPipeFaceFeatureExtractor:
    """Factory function to create the feature extractor."""
    return MediaPipeFaceFeatureExtractor(model_path)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "face_landmarker.task"
    extractor = create_extractor(model_path)
    print("Feature extractor created successfully")
    print(f"Feature names ({len(FacialFeatures.get_feature_names())}):")
    for name in FacialFeatures.get_feature_names():
        print(f"  {name}")
    extractor.close()