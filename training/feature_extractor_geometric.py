"""
Experimental Geometric Feature Extractor for FER2013

Adds expression-specific geometric features on top of the expanded MediaPipe features:
- Eye aspect ratios (left, right, average)
- Mouth aspect ratio, width, height, opening
- Eyebrow-eye distances (inner, outer)
- Eye distances (inter-ocular, inner corner)
- Nose-mouth distances
- Mouth-corner distances
- Facial symmetry measures
- Normalized landmark distances
- Relevant angles
- Head-pose features (already in expanded)

Normalization Strategy (same as expanded):
1. Translate landmarks so nose tip (landmark 1) is at origin
2. Scale by inter-ocular distance (landmarks 33 and 263) for scale invariance

Total features: ~1457 (expanded) + ~120 geometric = ~1577 features
"""

import math
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import sys

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
class GeometricFacialFeatures:
    """Container for geometric facial features from MediaPipe."""
    # All expanded features (1457)
    landmarks_normalized: List[float]  # 1404 values
    blendshapes: List[float]  # 52 values
    head_pitch: float
    head_yaw: float
    head_roll: float
    inter_ocular_distance: float
    detection_confidence: float
    
    # Geometric features (expression-specific)
    # Eye features
    eye_aspect_ratio_left: float
    eye_aspect_ratio_right: float
    eye_aspect_ratio_avg: float
    eye_openness_left: float
    eye_openness_right: float
    eye_openness_avg: float
    eye_squint_left: float
    eye_squint_right: float
    eye_wide_left: float
    eye_wide_right: float
    
    # Mouth features
    mouth_aspect_ratio: float
    mouth_width: float
    mouth_height: float
    mouth_openness: float
    mouth_upper_lip_height: float
    mouth_lower_lip_height: float
    mouth_corner_distance: float
    mouth_corner_upward_left: float
    mouth_corner_upward_right: float
    mouth_symmetry: float
    
    # Eyebrow features
    eyebrow_eye_distance_left_inner: float
    eyebrow_eye_distance_left_outer: float
    eyebrow_eye_distance_right_inner: float
    eyebrow_eye_distance_right_outer: float
    eyebrow_height_left: float
    eyebrow_height_right: float
    eyebrow_slope_left: float
    eyebrow_slope_right: float
    eyebrow_symmetry: float
    
    # Nose features
    nose_mouth_distance: float
    nose_width: float
    nose_tip_to_mouth_center: float
    
    # Face symmetry and proportions
    face_symmetry_vertical: float
    face_symmetry_horizontal: float
    face_aspect_ratio: float
    upper_face_height: float
    lower_face_height: float
    face_width: float
    
    # Angles
    mouth_corner_angle_left: float
    mouth_corner_angle_right: float
    eye_brow_angle_left: float
    eye_brow_angle_right: float
    
    # Normalized distances (scale invariant)
    normalized_eye_distance: float
    normalized_mouth_width: float
    normalized_mouth_height: float
    normalized_face_height: float
    
    def to_array(self) -> np.ndarray:
        """Convert to flat numpy array."""
        expanded_parts = [
            np.array(self.landmarks_normalized, dtype=np.float32),
            np.array(self.blendshapes, dtype=np.float32),
            np.array([self.head_pitch, self.head_yaw, self.head_roll], dtype=np.float32),
            np.array([self.inter_ocular_distance], dtype=np.float32),
            np.array([self.detection_confidence], dtype=np.float32),
        ]
        
        geometric_parts = [
            # Eye features
            np.array([self.eye_aspect_ratio_left, self.eye_aspect_ratio_right, 
                      self.eye_aspect_ratio_avg, self.eye_openness_left,
                      self.eye_openness_right, self.eye_openness_avg,
                      self.eye_squint_left, self.eye_squint_right,
                      self.eye_wide_left, self.eye_wide_right], dtype=np.float32),
            # Mouth features
            np.array([self.mouth_aspect_ratio, self.mouth_width, self.mouth_height,
                      self.mouth_openness, self.mouth_upper_lip_height,
                      self.mouth_lower_lip_height, self.mouth_corner_distance,
                      self.mouth_corner_upward_left, self.mouth_corner_upward_right,
                      self.mouth_symmetry], dtype=np.float32),
            # Eyebrow features
            np.array([self.eyebrow_eye_distance_left_inner, self.eyebrow_eye_distance_left_outer,
                      self.eyebrow_eye_distance_right_inner, self.eyebrow_eye_distance_right_outer,
                      self.eyebrow_height_left, self.eyebrow_height_right,
                      self.eyebrow_slope_left, self.eyebrow_slope_right,
                      self.eyebrow_symmetry], dtype=np.float32),
            # Nose features
            np.array([self.nose_mouth_distance, self.nose_width,
                      self.nose_tip_to_mouth_center], dtype=np.float32),
            # Face symmetry
            np.array([self.face_symmetry_vertical, self.face_symmetry_horizontal,
                      self.face_aspect_ratio, self.upper_face_height,
                      self.lower_face_height, self.face_width], dtype=np.float32),
            # Angles
            np.array([self.mouth_corner_angle_left, self.mouth_corner_angle_right,
                      self.eye_brow_angle_left, self.eye_brow_angle_right], dtype=np.float32),
            # Normalized distances
            np.array([self.normalized_eye_distance, self.normalized_mouth_width,
                      self.normalized_mouth_height, self.normalized_face_height], dtype=np.float32),
        ]
        
        return np.concatenate(expanded_parts + geometric_parts)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get ordered feature names."""
        names = []
        
        # Expanded feature names (1457)
        for i in range(468):
            names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
        blendshape_names = [
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
            "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
            "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
            "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
            "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
            "noseSneerLeft", "noseSneerRight"
        ]
        names.extend([f"bs_{name}" for name in blendshape_names])
        names.extend(["head_pitch", "head_yaw", "head_roll"])
        names.extend(["inter_ocular_distance", "detection_confidence"])
        
        # Geometric feature names
        names.extend([
            # Eye features
            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "eye_aspect_ratio_avg",
            "eye_openness_left", "eye_openness_right", "eye_openness_avg",
            "eye_squint_left", "eye_squint_right", "eye_wide_left", "eye_wide_right",
            # Mouth features
            "mouth_aspect_ratio", "mouth_width", "mouth_height", "mouth_openness",
            "mouth_upper_lip_height", "mouth_lower_lip_height", "mouth_corner_distance",
            "mouth_corner_upward_left", "mouth_corner_upward_right", "mouth_symmetry",
            # Eyebrow features
            "eyebrow_eye_distance_left_inner", "eyebrow_eye_distance_left_outer",
            "eyebrow_eye_distance_right_inner", "eyebrow_eye_distance_right_outer",
            "eyebrow_height_left", "eyebrow_height_right", "eyebrow_slope_left",
            "eyebrow_slope_right", "eyebrow_symmetry",
            # Nose features
            "nose_mouth_distance", "nose_width", "nose_tip_to_mouth_center",
            # Face symmetry
            "face_symmetry_vertical", "face_symmetry_horizontal", "face_aspect_ratio",
            "upper_face_height", "lower_face_height", "face_width",
            # Angles
            "mouth_corner_angle_left", "mouth_corner_angle_right",
            "eye_brow_angle_left", "eye_brow_angle_right",
            # Normalized distances
            "normalized_eye_distance", "normalized_mouth_width",
            "normalized_mouth_height", "normalized_face_height"
        ])
        return names
    
    @staticmethod
    def get_feature_groups() -> Dict[str, slice]:
        """Get slices for feature groups."""
        total_expanded = 1457
        total_geometric = 54  # Count of geometric features
        return {
            "expanded": slice(0, total_expanded),
            "geometric": slice(total_expanded, total_expanded + total_geometric),
            "eye": slice(total_expanded, total_expanded + 10),
            "mouth": slice(total_expanded + 10, total_expanded + 20),
            "eyebrow": slice(total_expanded + 20, total_expanded + 29),
            "nose": slice(total_expanded + 29, total_expanded + 32),
            "symmetry": slice(total_expanded + 32, total_expanded + 38),
            "angles": slice(total_expanded + 38, total_expanded + 42),
            "normalized": slice(total_expanded + 42, total_expanded + 46),
        }


class GeometricMediaPipeFeatureExtractor:
    """
    MediaPipe-based feature extractor with expression-specific geometric features.
    
    Extracts expanded features + geometric features for expression recognition.
    """
    
    # MediaPipe blendshape names (52 standard face blendshapes)
    BLENDSHAPE_NAMES = [
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
        "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
        "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
        "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
        "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
        "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
        "noseSneerLeft", "noseSneerRight"
    ]
    
    # Key landmark indices (MediaPipe 468 landmarks)
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    
    # Eye landmarks
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263
    
    # Eyebrow landmarks
    LEFT_BROW_INNER = 70
    LEFT_BROW_OUTER = 105
    RIGHT_BROW_INNER = 300
    RIGHT_BROW_OUTER = 334
    LEFT_BROW_TOP = 107
    RIGHT_BROW_TOP = 336
    
    # Mouth landmarks
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    UPPER_LIP_TOP = 12
    UPPER_LIP_BOTTOM = 15
    LOWER_LIP_TOP = 16
    LOWER_LIP_BOTTOM = 17
    
    # Nose landmarks
    NOSE_TIP = 1
    NOSE_LEFT = 98
    NOSE_RIGHT = 327
    NOSE_BRIDGE = 6
    
    # Eyebrow top landmarks
    LEFT_BROW_TOP = 107
    RIGHT_BROW_TOP = 336
    
    def __init__(self, model_path: str, normalize_landmarks: bool = True):
        """
        Initialize MediaPipe FaceLandmarker.
        
        Args:
            model_path: Path to face_landmarker.task model file
            normalize_landmarks: If True, normalize landmarks for translation/scale invariance
        """
        self.normalize_landmarks = normalize_landmarks
        
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
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[GeometricFacialFeatures], bool]:
        """
        Extract geometric facial features from a single image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Tuple of (GeometricFacialFeatures or None, face_detected_flag)
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
        
        # Extract raw landmarks (468 * 3)
        landmarks_raw = np.zeros((468, 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks):
            if i < 468:
                landmarks_raw[i, 0] = lm.x
                landmarks_raw[i, 1] = lm.y
                landmarks_raw[i, 2] = lm.z
        
        # Check for invalid landmarks
        if np.any(np.isnan(landmarks_raw)) or np.any(np.isinf(landmarks_raw)):
            return None, False
        
        # Calculate inter-ocular distance (for normalization)
        left_eye = landmarks_raw[self.LEFT_EYE_OUTER]
        right_eye = landmarks_raw[self.RIGHT_EYE_OUTER]
        inter_ocular_dist = np.linalg.norm(left_eye - right_eye)
        
        if inter_ocular_dist < 1e-6:
            return None, False
        
        # Normalize landmarks if requested
        if self.normalize_landmarks:
            landmarks_norm = self._normalize_landmarks(landmarks_raw, inter_ocular_dist)
        else:
            landmarks_norm = landmarks_raw
        
        # Flatten landmarks
        landmarks_flat = landmarks_norm.flatten()
        
        # Extract blendshapes (52)
        blendshapes = np.zeros(52, dtype=np.float32)
        if result.face_blendshapes:
            for i, blendshape in enumerate(result.face_blendshapes[0]):
                if i < 52:
                    blendshapes[i] = float(blendshape.score)
        
        # Calculate head pose
        head_pose = self._calculate_head_pose(face_landmarks, w, h)
        
        # Compute geometric features
        geometric_features = self._compute_geometric_features(landmarks_raw, landmarks_norm, inter_ocular_dist, w, h)
        
        features = GeometricFacialFeatures(
            landmarks_normalized=landmarks_flat.tolist(),
            blendshapes=blendshapes.tolist(),
            head_pitch=head_pose.get("pitch", 0.0),
            head_yaw=head_pose.get("yaw", 0.0),
            head_roll=head_pose.get("roll", 0.0),
            inter_ocular_distance=float(inter_ocular_dist),
            detection_confidence=detection_confidence,
            **geometric_features
        )
        
        return features, True
    
    def _normalize_landmarks(self, landmarks: np.ndarray, inter_ocular_dist: float) -> np.ndarray:
        """Normalize landmarks for translation and scale invariance."""
        nose_tip = landmarks[self.NOSE_TIP].copy()
        landmarks_centered = landmarks - nose_tip
        landmarks_scaled = landmarks_centered / inter_ocular_dist
        return landmarks_scaled
    
    def _calculate_head_pose(self, face_landmarks, w: int, h: int) -> Dict[str, float]:
        """Calculate head pose (pitch, yaw, roll) from facial landmarks."""
        try:
            nose_tip = face_landmarks[self.NOSE_TIP]
            left_eye = face_landmarks[self.LEFT_EYE_OUTER]
            right_eye = face_landmarks[self.RIGHT_EYE_OUTER]
            chin = face_landmarks[self.CHIN]
            
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
    
    def _compute_geometric_features(self, landmarks_raw: np.ndarray, landmarks_norm: np.ndarray, 
                                     inter_ocular_dist: float, w: int, h: int) -> Dict[str, float]:
        """Compute all geometric features from landmarks."""
        features = {}
        
        # Helper: get 2D point
        def pt2d(idx):
            return landmarks_raw[idx][:2]
        
        def pt3d(idx):
            return landmarks_raw[idx]
        
        def dist2d(i, j):
            return np.linalg.norm(pt2d(i) - pt2d(j))
        
        def dist3d(i, j):
            return np.linalg.norm(landmarks_raw[i] - landmarks_raw[j])
        
        def angle(p1, p2, p3):
            """Angle at p2 formed by p1-p2-p3."""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        # ===== EYE FEATURES =====
        # Eye aspect ratio (EAR) = (vertical distances) / (horizontal distance)
        # Left eye
        left_eye_vert1 = dist2d(self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM)
        left_eye_horiz = dist2d(self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT)
        features["eye_aspect_ratio_left"] = float(left_eye_vert1 / (left_eye_horiz + 1e-8))
        
        # Right eye
        right_eye_vert1 = dist2d(self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM)
        right_eye_horiz = dist2d(self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT)
        features["eye_aspect_ratio_right"] = float(right_eye_vert1 / (right_eye_horiz + 1e-8))
        
        features["eye_aspect_ratio_avg"] = float((features["eye_aspect_ratio_left"] + features["eye_aspect_ratio_right"]) / 2)
        
        # Eye openness (normalized vertical opening)
        features["eye_openness_left"] = float(left_eye_vert1 / (inter_ocular_dist + 1e-8))
        features["eye_openness_right"] = float(right_eye_vert1 / (inter_ocular_dist + 1e-8))
        features["eye_openness_avg"] = float((features["eye_openness_left"] + features["eye_openness_right"]) / 2)
        
        # Eye squint (inverse of openness)
        features["eye_squint_left"] = float(1.0 - features["eye_openness_left"])
        features["eye_squint_right"] = float(1.0 - features["eye_openness_right"])
        
        # Eye wide (horizontal opening)
        features["eye_wide_left"] = float(left_eye_horiz / (inter_ocular_dist + 1e-8))
        features["eye_wide_right"] = float(right_eye_horiz / (inter_ocular_dist + 1e-8))
        
        # ===== MOUTH FEATURES =====
        mouth_width = dist2d(self.MOUTH_LEFT, self.MOUTH_RIGHT)
        mouth_height = dist2d(self.MOUTH_TOP, self.MOUTH_BOTTOM)
        
        features["mouth_aspect_ratio"] = float(mouth_height / (mouth_width + 1e-8))
        features["mouth_width"] = float(mouth_width / (inter_ocular_dist + 1e-8))
        features["mouth_height"] = float(mouth_height / (inter_ocular_dist + 1e-8))
        features["mouth_openness"] = float(mouth_height / (inter_ocular_dist + 1e-8))
        
        # Upper/lower lip height
        upper_lip_height = dist2d(self.UPPER_LIP_TOP, self.UPPER_LIP_BOTTOM)
        lower_lip_height = dist2d(self.LOWER_LIP_TOP, self.LOWER_LIP_BOTTOM)
        features["mouth_upper_lip_height"] = float(upper_lip_height / (inter_ocular_dist + 1e-8))
        features["mouth_lower_lip_height"] = float(lower_lip_height / (inter_ocular_dist + 1e-8))
        
        # Mouth corner distance
        features["mouth_corner_distance"] = float(mouth_width / (inter_ocular_dist + 1e-8))
        
        # Mouth corner upward (smile indicator)
        mouth_center_y = (landmarks_raw[self.MOUTH_TOP][1] + landmarks_raw[self.MOUTH_BOTTOM][1]) / 2
        left_corner_up = float((mouth_center_y - landmarks_raw[self.MOUTH_LEFT][1]) / (inter_ocular_dist + 1e-8))
        right_corner_up = float((mouth_center_y - landmarks_raw[self.MOUTH_RIGHT][1]) / (inter_ocular_dist + 1e-8))
        features["mouth_corner_upward_left"] = left_corner_up
        features["mouth_corner_upward_right"] = right_corner_up
        
        # Mouth symmetry
        left_corner_to_center = dist2d(self.MOUTH_LEFT, self.NOSE_TIP)
        right_corner_to_center = dist2d(self.MOUTH_RIGHT, self.NOSE_TIP)
        features["mouth_symmetry"] = float(1.0 - abs(left_corner_to_center - right_corner_to_center) / (left_corner_to_center + right_corner_to_center + 1e-8))
        
        # ===== EYEBROW FEATURES =====
        # Eyebrow-eye distances
        features["eyebrow_eye_distance_left_inner"] = float(dist2d(self.LEFT_BROW_INNER, self.LEFT_EYE_INNER) / (inter_ocular_dist + 1e-8))
        features["eyebrow_eye_distance_left_outer"] = float(dist2d(self.LEFT_BROW_OUTER, self.LEFT_EYE_OUTER) / (inter_ocular_dist + 1e-8))
        features["eyebrow_eye_distance_right_inner"] = float(dist2d(self.RIGHT_BROW_INNER, self.RIGHT_EYE_INNER) / (inter_ocular_dist + 1e-8))
        features["eyebrow_eye_distance_right_outer"] = float(dist2d(self.RIGHT_BROW_OUTER, self.RIGHT_EYE_OUTER) / (inter_ocular_dist + 1e-8))
        
        # Eyebrow height (distance from brow to eye)
        features["eyebrow_height_left"] = float(dist2d(self.LEFT_BROW_TOP, self.LEFT_EYE_TOP) / (inter_ocular_dist + 1e-8))
        features["eyebrow_height_right"] = float(dist2d(self.RIGHT_BROW_TOP, self.RIGHT_EYE_TOP) / (inter_ocular_dist + 1e-8))
        
        # Eyebrow slope (angle of brow)
        features["eyebrow_slope_left"] = float(angle(pt3d(self.LEFT_BROW_INNER), pt3d(self.LEFT_BROW_OUTER), pt3d(self.NOSE_TIP)))
        features["eyebrow_slope_right"] = float(angle(pt3d(self.RIGHT_BROW_OUTER), pt3d(self.RIGHT_BROW_INNER), pt3d(self.NOSE_TIP)))
        
        # Eyebrow symmetry
        left_brow_height = dist2d(self.LEFT_BROW_TOP, self.LEFT_EYE_TOP)
        right_brow_height = dist2d(self.RIGHT_BROW_TOP, self.RIGHT_EYE_TOP)
        features["eyebrow_symmetry"] = float(1.0 - abs(left_brow_height - right_brow_height) / (left_brow_height + right_brow_height + 1e-8))
        
# Compute mouth center as midpoint of mouth corners
        mouth_center = (landmarks_raw[self.MOUTH_LEFT] + landmarks_raw[self.MOUTH_RIGHT]) / 2
        
        # ===== NOSE FEATURES =====
        features["nose_mouth_distance"] = float(np.linalg.norm(landmarks_raw[self.NOSE_TIP] - mouth_center) / (inter_ocular_dist + 1e-8))
        features["nose_width"] = float(dist2d(self.NOSE_LEFT, self.NOSE_RIGHT) / (inter_ocular_dist + 1e-8))
        features["nose_tip_to_mouth_center"] = float(np.linalg.norm(landmarks_raw[self.NOSE_TIP] - mouth_center) / (inter_ocular_dist + 1e-8))
        
        # ===== FACE SYMMETRY AND PROPORTIONS =====
        face_width = dist2d(self.LEFT_EAR, self.RIGHT_EAR)
        face_height = dist2d(self.NOSE_TIP, self.CHIN)
        
        features["face_symmetry_vertical"] = float(1.0 - abs(dist2d(self.LEFT_EAR, self.NOSE_TIP) - dist2d(self.RIGHT_EAR, self.NOSE_TIP)) / (inter_ocular_dist + 1e-8))
        features["face_symmetry_horizontal"] = float(1.0 - abs(dist2d(self.LEFT_EAR, self.CHIN) - dist2d(self.RIGHT_EAR, self.CHIN)) / (face_width + 1e-8))
        features["face_aspect_ratio"] = float(face_height / (face_width + 1e-8))
        
        upper_face_height = dist2d(self.NOSE_TIP, self.LEFT_EYE_TOP)
        lower_face_height = dist2d(self.NOSE_TIP, self.CHIN)
        features["upper_face_height"] = float(upper_face_height / (inter_ocular_dist + 1e-8))
        features["lower_face_height"] = float(lower_face_height / (inter_ocular_dist + 1e-8))
        features["face_width"] = float(face_width / (inter_ocular_dist + 1e-8))
        
        # ===== ANGLES =====
        # Mouth corner angles
        features["mouth_corner_angle_left"] = float(angle(pt3d(self.LEFT_BROW_INNER), pt3d(self.MOUTH_LEFT), mouth_center))
        features["mouth_corner_angle_right"] = float(angle(pt3d(self.RIGHT_BROW_INNER), pt3d(self.MOUTH_RIGHT), mouth_center))
        
        # Eye brow angles
        features["eye_brow_angle_left"] = float(angle(pt3d(self.LEFT_BROW_INNER), pt3d(self.LEFT_BROW_OUTER), pt3d(self.LEFT_EYE_OUTER)))
        features["eye_brow_angle_right"] = float(angle(pt3d(self.RIGHT_BROW_OUTER), pt3d(self.RIGHT_BROW_INNER), pt3d(self.RIGHT_EYE_OUTER)))
        
        # ===== NORMALIZED DISTANCES =====
        features["normalized_eye_distance"] = float(dist2d(self.LEFT_EYE_INNER, self.RIGHT_EYE_INNER) / (inter_ocular_dist + 1e-8))
        features["normalized_mouth_width"] = float(mouth_width / (inter_ocular_dist + 1e-8))
        features["normalized_mouth_height"] = float(mouth_height / (inter_ocular_dist + 1e-8))
        features["normalized_face_height"] = float(face_height / (inter_ocular_dist + 1e-8))
        
        return features
    
    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


def create_geometric_extractor(model_path: str = "face_landmarker.task", 
                               normalize_landmarks: bool = True) -> GeometricMediaPipeFeatureExtractor:
    """Factory function to create the geometric feature extractor."""
    return GeometricMediaPipeFeatureExtractor(model_path, normalize_landmarks)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "face_landmarker.task"
    extractor = create_geometric_extractor(model_path)
    print("Geometric feature extractor created successfully")
    names = GeometricFacialFeatures.get_feature_names()
    print(f"Total features: {len(names)}")
    groups = GeometricFacialFeatures.get_feature_groups()
    for group_name, slice_obj in groups.items():
        print(f"  {group_name}: {slice_obj.stop - slice_obj.start} features")
    extractor.close()