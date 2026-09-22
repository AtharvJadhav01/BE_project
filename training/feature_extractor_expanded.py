"""
Expanded MediaPipe Feature Extractor for FER2013

Extracts comprehensive facial features from MediaPipe FaceLandmarker:
- 468 3D landmarks (x, y, z) = 1404 raw coordinates
- 52 blendshape coefficients
- Head pose (pitch, yaw, roll) = 3 values
- Normalized landmark geometry (translation/scale invariant)

Normalization Strategy:
1. Translate landmarks so nose tip (landmark 1) is at origin
2. Scale by inter-ocular distance (landmarks 33 and 263) for scale invariance
3. Optionally rotate to canonical pose using head pose estimation

This reduces dependence on face position, camera distance, and image translation.
"""

import math
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
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
class ExpandedFacialFeatures:
    """Container for expanded facial features from MediaPipe."""
    # Raw landmarks (468 * 3 = 1404 values) - normalized
    landmarks_normalized: List[float]  # Flattened [x1, y1, z1, x2, y2, z2, ...]
    
    # Blendshapes (52 coefficients)
    blendshapes: List[float]  # 52 values in MediaPipe order
    
    # Head pose (3 values)
    head_pitch: float
    head_yaw: float
    head_roll: float
    
    # Inter-ocular distance (for reference)
    inter_ocular_distance: float
    
    # Face detection confidence
    detection_confidence: float
    
    def to_array(self) -> np.ndarray:
        """Convert to flat numpy array."""
        parts = [
            np.array(self.landmarks_normalized, dtype=np.float32),
            np.array(self.blendshapes, dtype=np.float32),
            np.array([self.head_pitch, self.head_yaw, self.head_roll], dtype=np.float32),
            np.array([self.inter_ocular_distance], dtype=np.float32),
            np.array([self.detection_confidence], dtype=np.float32),
        ]
        return np.concatenate(parts)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get ordered feature names."""
        names = []
        # Landmark names
        for i in range(468):
            names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
        # Blendshape names (MediaPipe standard 52)
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
        # Head pose
        names.extend(["head_pitch", "head_yaw", "head_roll"])
        # Metadata
        names.extend(["inter_ocular_distance", "detection_confidence"])
        return names
    
    @staticmethod
    def get_feature_groups() -> Dict[str, slice]:
        """Get slices for feature groups for importance analysis."""
        total_landmarks = 468 * 3
        total_blendshapes = 52
        return {
            "landmarks": slice(0, total_landmarks),
            "blendshapes": slice(total_landmarks, total_landmarks + total_blendshapes),
            "head_pose": slice(total_landmarks + total_blendshapes, total_landmarks + total_blendshapes + 3),
            "metadata": slice(total_landmarks + total_blendshapes + 3, total_landmarks + total_blendshapes + 5),
        }


class ExpandedMediaPipeFeatureExtractor:
    """
    Expanded MediaPipe-based feature extractor for static images.
    
    Extracts comprehensive features including raw landmarks, blendshapes,
    head pose, and normalized geometry.
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
    
    # Key landmark indices
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    
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
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[ExpandedFacialFeatures], bool]:
        """
        Extract expanded facial features from a single image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Tuple of (ExpandedFacialFeatures or None, face_detected_flag)
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
        
        features = ExpandedFacialFeatures(
            landmarks_normalized=landmarks_flat.tolist(),
            blendshapes=blendshapes.tolist(),
            head_pitch=head_pose.get("pitch", 0.0),
            head_yaw=head_pose.get("yaw", 0.0),
            head_roll=head_pose.get("roll", 0.0),
            inter_ocular_distance=float(inter_ocular_dist),
            detection_confidence=detection_confidence
        )
        
        return features, True
    
    def _normalize_landmarks(self, landmarks: np.ndarray, inter_ocular_dist: float) -> np.ndarray:
        """
        Normalize landmarks for translation and scale invariance.
        
        Normalization steps:
        1. Translate so nose tip (landmark 1) is at origin
        2. Scale by inter-ocular distance (distance between eye outer corners)
        
        This makes features invariant to:
        - Face position in image (translation)
        - Camera distance / face size (scale)
        - Image resolution (scale)
        
        Note: We do NOT rotate to canonical pose because head pose itself
        carries discriminative information for expression recognition.
        
        Args:
            landmarks: (468, 3) array of raw landmark coordinates
            inter_ocular_dist: Distance between left and right eye outer corners
            
        Returns:
            Normalized landmarks (468, 3)
        """
        # Step 1: Translate to nose tip origin
        nose_tip = landmarks[self.NOSE_TIP].copy()
        landmarks_centered = landmarks - nose_tip
        
        # Step 2: Scale by inter-ocular distance
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
    
    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


def create_expanded_extractor(model_path: str = "face_landmarker.task", 
                              normalize_landmarks: bool = True) -> ExpandedMediaPipeFeatureExtractor:
    """Factory function to create the expanded feature extractor."""
    return ExpandedMediaPipeFeatureExtractor(model_path, normalize_landmarks)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "face_landmarker.task"
    extractor = create_expanded_extractor(model_path)
    print("Expanded feature extractor created successfully")
    names = ExpandedFacialFeatures.get_feature_names()
    print(f"Total features: {len(names)}")
    groups = ExpandedFacialFeatures.get_feature_groups()
    for group_name, slice_obj in groups.items():
        print(f"  {group_name}: {slice_obj.stop - slice_obj.start} features")
    extractor.close()