"""
Face feature extraction for converting MediaPipe FaceLandmarker output 
into meaningful numerical features for analysis.
"""

import math
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from analysis.interfaces import FeatureExtractor, FeatureVector, DataQuality
from analysis.results import FeatureVector as FeatureVectorModel
from analysis.features import BaseFeatureExtractor, compute_statistical_features, compute_rate_features


@dataclass
class FacialFeatures:
    """Container for extracted facial features."""
    # Eye features
    eye_openness_left: float = 0.0
    eye_openness_right: float = 0.0
    eye_openness_avg: float = 0.0
    eye_aspect_ratio_left: float = 0.0
    eye_aspect_ratio_right: float = 0.0
    blink_rate: float = 0.0  # blinks per minute
    
    # Mouth features
    mouth_openness: float = 0.0
    mouth_width: float = 0.0  # smile indicator
    lip_press: float = 0.0
    smile_asymmetry: float = 0.0
    # Phase 1 additions
    mouth_funnel: float = 0.0
    mouth_pucker: float = 0.0
    mouth_stretch_left: float = 0.0
    mouth_stretch_right: float = 0.0
    
    # Eyebrow features
    brow_inner_up: float = 0.0  # surprise
    brow_down_left: float = 0.0   # anger/frown
    brow_down_right: float = 0.0  # anger/frown
    brow_outer_up_left: float = 0.0
    brow_outer_up_right: float = 0.0
    
    # Head pose
    head_pitch: float = 0.0  # degrees
    head_yaw: float = 0.0    # degrees
    head_roll: float = 0.0   # degrees
    
    # Facial movement and dynamics
    facial_movement_velocity: float = 0.0
    facial_symmetry: float = 1.0  # 0-1, 1 = perfectly symmetric
    
    # Overall expressivity
    overall_expressivity: float = 0.0
    
    # Nose features for emotion estimation
    nose_sneer_left: float = 0.0
    nose_sneer_right: float = 0.0


class FaceFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts interpretable facial features from MediaPipe FaceLandmarker output.
    
    Converts 468 landmarks and 52 blendshapes into meaningful facial action units
    and geometric features that can be used for state estimation.
    """
    
    # MediaPipe blendshape indices for key facial actions
    BLENDSHAPE_INDICES = {
        # Eyes
        "eyeBlinkLeft": 4,
        "eyeBlinkRight": 5,
        "eyeLookDownLeft": 6,
        "eyeLookDownRight": 7,
        "eyeLookInLeft": 8,
        "eyeLookInRight": 9,
        "eyeLookOutLeft": 10,
        "eyeLookOutRight": 11,
        "eyeLookUpLeft": 12,
        "eyeLookUpRight": 13,
        "eyeSquintLeft": 14,
        "eyeSquintRight": 15,
        "eyeWideLeft": 16,
        "eyeWideRight": 17,
        
        # Eyebrows
        "browDownLeft": 0,
        "browDownRight": 1,
        "browInnerUp": 2,
        "browOuterUpLeft": 3,
        "browOuterUpRight": 4,
        
        # Mouth
        "mouthClose": 28,
        "mouthFrownLeft": 30,
        "mouthFrownRight": 31,
        "mouthFunnel": 32,
        "mouthLeft": 33,
        "mouthPucker": 37,
        "mouthRight": 38,
        "mouthShrugLower": 39,
        "mouthShrugUpper": 40,
        "mouthSmileLeft": 41,
        "mouthSmileRight": 42,
        "mouthStretchLeft": 43,
        "mouthStretchRight": 44,
        "mouthUpperUpLeft": 45,
        "mouthUpperUpRight": 46,
        "mouthDimpleLeft": 29,
        "mouthDimpleRight": 34,
        "mouthPressLeft": 35,
        "mouthPressRight": 36,
        "mouthRollLower": 47,
        "mouthRollUpper": 48,
        "mouthShrugLower": 39,
        "mouthShrugUpper": 40,
        "mouthDimpleLeft": 29,
        "mouthDimpleRight": 34,
        
        # Jaw
        "jawOpen": 18,
        "jawForward": 19,
        "jawLeft": 20,
        "jawRight": 21,
        
        # Nose
        "noseSneerLeft": 50,
        "noseSneerRight": 51
    }
    
    # Key landmark indices for geometric calculations
    LANDMARK_INDICES = {
        # Eyes - for eye aspect ratio (EAR)
        "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 162, 246],
        "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        
        # Mouth - for mouth aspect ratio (MAR)
        "mouth": [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        
        # Face outline - for symmetry
        "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
        
        # Head pose
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_ear": 234,
        "right_ear": 454
    }
    
    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self._blink_state_history: List[bool] = []
        self._blink_timestamps: List[float] = []
        self._last_landmarks: Optional[List[Dict[str, float]]] = None
        self._last_timestamp: Optional[float] = None
        
        # Configuration parameters
        self.eye_aspect_ratio_threshold = config.get("eye_aspect_ratio_threshold", 0.2)
        self.blink_rate_window_sec = config.get("blink_rate_window_sec", 10.0)
        self.facial_movement_history_size = config.get("facial_movement_history_size", 10)
        
    def _extract_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract facial features from MediaPipe FaceDetectionResult.
        
        Args:
            raw_data: Dictionary containing face detection result with:
                     - landmarks_3d: List of 468 3D landmarks
                     - blendshapes: Dict of 52 blendshape coefficients
                     - head_pose: Dict with pitch, yaw, roll
                     - iris: Dict with left/right iris positions
                     
        Returns:
            Dictionary of facial features
        """
        # Handle case where no face is detected
        if not raw_data or not raw_data.get("face_detected", False):
            return self._get_default_features()
            
        landmarks_3d = raw_data.get("landmarks_3d", [])
        blendshapes = raw_data.get("blendshapes", {})
        head_pose = raw_data.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
        
        if not landmarks_3d:
            return self._get_default_features()
            
        features = FacialFeatures()
        
        # Extract eye features
        self._extract_eye_features(landmarks_3d, blendshapes, features)
        
        # Extract mouth features
        self._extract_mouth_features(landmarks_3d, blendshapes, features)
        
        # Extract eyebrow features
        self._extract_eyebrow_features(blendshapes, features)
        
        # Extract head pose
        features.head_pitch = head_pose.get("pitch", 0.0)
        features.head_yaw = head_pose.get("yaw", 0.0)
        features.head_roll = head_pose.get("roll", 0.0)
        
        # Extract facial movement and dynamics
        current_timestamp = time.monotonic()
        self._extract_facial_dynamics(landmarks_3d, current_timestamp, features)
        
        # Calculate overall expressivity
        features.overall_expressivity = self._calculate_expressivity(features)
        
        # Convert to dictionary
        return self._features_to_dict(features)

    def extract(self, raw_data: Any, timestamp: float | None = None, wallclock: str | None = None) -> Any:
        """
        Override extract to handle no-face case with proper quality.
        """
        # Handle case where no face is detected
        if not raw_data or not raw_data.get("face_detected", False):
            start_time = time.perf_counter()
            features = self._get_default_features()
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            return self._create_feature_vector(
                features=features,
                timestamp=timestamp,
                wallclock=wallclock,
                quality=DataQuality.INSUFFICIENT,
                processing_time_ms=processing_time_ms
            )
        
        # Use parent implementation for face detected case
        return super().extract(raw_data, timestamp, wallclock)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no face is detected."""
        features = FacialFeatures()
        return self._features_to_dict(features)
    
    def _features_to_dict(self, features: FacialFeatures) -> Dict[str, float]:
        """Convert FacialFeatures dataclass to dictionary."""
        return {
            # Eye features
            "eye_openness_left": features.eye_openness_left,
            "eye_openness_right": features.eye_openness_right,
            "eye_openness_avg": features.eye_openness_avg,
            "eye_aspect_ratio_left": features.eye_aspect_ratio_left,
            "eye_aspect_ratio_right": features.eye_aspect_ratio_right,
            "blink_rate": features.blink_rate,
            
            # Mouth features
            "mouth_openness": features.mouth_openness,
            "mouth_width": features.mouth_width,
            "lip_press": features.lip_press,
            "smile_asymmetry": features.smile_asymmetry,
            # Phase 1 additions
            "mouth_funnel": features.mouth_funnel,
            "mouth_pucker": features.mouth_pucker,
            "mouth_stretch_left": features.mouth_stretch_left,
            "mouth_stretch_right": features.mouth_stretch_right,
            
            # Eyebrow features
            "brow_inner_up": features.brow_inner_up,
            "brow_down_left": features.brow_down_left,
            "brow_down_right": features.brow_down_right,
            "brow_outer_up_left": features.brow_outer_up_left,
            "brow_outer_up_right": features.brow_outer_up_right,
            
            # Head pose
            "head_pitch": features.head_pitch,
            "head_yaw": features.head_yaw,
            "head_roll": features.head_roll,
            
            # Facial movement
            "facial_movement_velocity": features.facial_movement_velocity,
            "facial_symmetry": features.facial_symmetry,
            
            # Overall
            "overall_expressivity": features.overall_expressivity,
            
            # Nose features
            "nose_sneer_left": features.nose_sneer_left,
            "nose_sneer_right": features.nose_sneer_right
        }
    
    def _extract_eye_features(self, landmarks_3d: List[Dict[str, float]], 
                            blendshapes: Dict[str, float], 
                            features: FacialFeatures) -> None:
        """Extract eye-related features."""
        try:
            # Eye Aspect Ratio (EAR) - indicates eye openness
            # Using landmarks for left and right eye
            left_ear = self._calculate_eye_aspect_ratio(landmarks_3d, "left")
            right_ear = self._calculate_eye_aspect_ratio(landmarks_3d, "right")
            
            features.eye_aspect_ratio_left = left_ear
            features.eye_aspect_ratio_right = right_ear
            
            # Convert EAR to openness (higher EAR = more open)
            # Normalized to 0-1 range where 0 = closed, 1 = fully open
            features.eye_openness_left = max(0.0, min(1.0, (left_ear - 0.2) / 0.3))
            features.eye_openness_right = max(0.0, min(1.0, (right_ear - 0.2) / 0.3))
            features.eye_openness_avg = (features.eye_openness_left + features.eye_openness_right) / 2.0
            
            # Blink detection from blendshapes
            left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
            right_blink = blendshapes.get("eyeBlinkRight", 0.0)
            blink_value = max(left_blink, right_blink)
            
            # Track blink state for rate calculation
            is_blinking = blink_value > self.eye_aspect_ratio_threshold
            self._blink_state_history.append(is_blinking)
            current_time = time.monotonic()
            self._blink_timestamps.append(current_time)
            
            # Remove old entries outside the window
            cutoff_time = current_time - self.blink_rate_window_sec
            while self._blink_timestamps and self._blink_timestamps[0] < cutoff_time:
                self._blink_timestamps.pop(0)
                self._blink_state_history.pop(0)
                
            # Calculate blink rate (blinks per minute)
            if len(self._blink_timestamps) >= 2:
                # Count transitions from non-blink to blink
                blink_count = 0
                for i in range(1, len(self._blink_state_history)):
                    if not self._blink_state_history[i-1] and self._blink_state_history[i]:
                        blink_count += 1
                
                time_span = self._blink_timestamps[-1] - self._blink_timestamps[0]
                if time_span > 0:
                    features.blink_rate = (blink_count / time_span) * 60.0
                    
        except Exception:
            # Default values on error
            features.eye_openness_left = 0.0
            features.eye_openness_right = 0.0
            features.eye_openness_avg = 0.0
            features.eye_aspect_ratio_left = 0.0
            features.eye_aspect_ratio_right = 0.0
            features.blink_rate = 0.0
    
    def _calculate_eye_aspect_ratio(self, landmarks_3d: List[Dict[str, float]], eye: str) -> float:
        """Calculate Eye Aspect Ratio (EAR) for given eye."""
        # Using 6-point EAR formula
        if eye == "left":
            # Left eye landmarks (approximate indices)
            points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 162]
        else:  # right
            # Right eye landmarks (approximate indices)
            points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
        if len(landmarks_3d) < max(points) + 1:
            return 0.3  # Default closed eye value
            
        try:
            # Get key points for EAR calculation
            # Using simplified 6-point model: P1-P6
            p1 = landmarks_3d[points[0]]  # corner
            p2 = landmarks_3d[points[1]]  # top
            p3 = landmarks_3d[points[2]]  # top
            p4 = landmarks_3d[points[3]]  # corner
            p5 = landmarks_3d[points[4]]  # bottom
            p6 = landmarks_3d[points[5]]  # bottom
            
            # Calculate vertical distances
            vert1 = self._euclidean_distance(p2, p6)
            vert2 = self._euclidean_distance(p3, p5)
            
            # Calculate horizontal distance
            horiz = self._euclidean_distance(p1, p4)
            
            # Avoid division by zero
            if horiz == 0:
                return 0.0
                
            ear = (vert1 + vert2) / (2.0 * horiz)
            return ear
        except Exception:
            return 0.3  # Default value
    
    def _euclidean_distance(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return math.sqrt(
            (p1["x"] - p2["x"])**2 + 
            (p1["y"] - p2["y"])**2 + 
            (p1["z"] - p2["z"])**2
        )
    
    def _extract_mouth_features(self, landmarks_3d: List[Dict[str, float]], 
                               blendshapes: Dict[str, float], 
                               features: FacialFeatures) -> None:
        """Extract mouth-related features."""
        try:
            # Mouth openness from blendshapes
            features.mouth_openness = blendshapes.get("jawOpen", 0.0)
            
            # Smile indicator (average of left and right smile blendshapes)
            smile_left = blendshapes.get("mouthSmileLeft", 0.0)
            smile_right = blendshapes.get("mouthSmileRight", 0.0)
            features.mouth_width = (smile_left + smile_right) / 2.0
            
            # Lip press (pressing lips together)
            features.lip_press = blendshapes.get("mouthPressLeft", 0.0) + blendshapes.get("mouthPressRight", 0.0)
            
            # Smile asymmetry (difference between left and right smile)
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
            features.brow_outer_up_right = 0.0
    
    def _extract_facial_dynamics(self, landmarks_3d: List[Dict[str, float]], 
                               current_timestamp: float, 
                               features: FacialFeatures) -> None:
        """Extract facial movement and dynamics features."""
        try:
            # Calculate facial movement velocity compared to last frame
            if (self._last_landmarks is not None and 
                self._last_timestamp is not None and
                len(landmarks_3d) == len(self._last_landmarks)):
                
                # Calculate average landmark movement
                total_movement = 0.0
                valid_points = 0
                
                # Use a subset of landmarks for performance (every 10th landmark)
                sample_indices = list(range(0, len(landmarks_3d), 10))
                
                for idx in sample_indices:
                    if idx < len(landmarks_3d) and idx < len(self._last_landmarks):
                        curr_point = landmarks_3d[idx]
                        last_point = self._last_landmarks[idx]
                        
                        movement = self._euclidean_distance(curr_point, last_point)
                        total_movement += movement
                        valid_points += 1
                
                if valid_points > 0:
                    avg_movement = total_movement / valid_points
                    time_delta = current_timestamp - self._last_timestamp
                    
                    if time_delta > 0:
                        features.facial_movement_velocity = avg_movement / time_delta  # units per second
            
            # Calculate facial symmetry
            features.facial_symmetry = self._calculate_facial_symmetry(landmarks_3d)
            
            # Store current state for next comparison
            self._last_landmarks = [lm.copy() for lm in landmarks_3d]  # Deep copy
            self._last_timestamp = current_timestamp
            
        except Exception:
            features.facial_movement_velocity = 0.0
            features.facial_symmetry = 1.0
    
    def _calculate_facial_symmetry(self, landmarks_3d: List[Dict[str, float]]) -> float:
        """Calculate facial symmetry score (0-1, where 1 is perfectly symmetric)."""
        try:
            # Compare left and right sides of face
            # Using symmetrical landmark pairs
            
            # Define symmetrical point pairs (approximate MediaPipe indices)
            symmetrical_pairs = [
                (33, 263),   # eyes outer corners
                (133, 362),  # eyes inner corners
                (70, 300),   # eyebrows
                (105, 334),  # more eyebrows
                (61, 291),   # mouth corners
                (78, 308),   # more mouth
                (1, 1),      # nose tip (center)
                (199, 199),  # chin area (center)
            ]
            
            total_asymmetry = 0.0
            valid_pairs = 0
            
            face_width = self._get_face_width(landmarks_3d)
            
            if face_width > 0:
                for left_idx, right_idx in symmetrical_pairs:
                    if (left_idx < len(landmarks_3d) and 
                        right_idx < len(landmarks_3d)):
                        
                        left_point = landmarks_3d[left_idx]
                        right_point = landmarks_3d[right_idx]
                        
                        # Calculate distance from midline
                        # Assuming face center is at x=0.5 in normalized coordinates
                        midline_x = 0.5
                        left_distance = abs(left_point["x"] - midline_x)
                        right_distance = abs(right_point["x"] - midline_x)
                        
                        # Asymmetry is difference in distances from midline
                        asymmetry = abs(left_distance - right_distance)
                        normalized_asymmetry = asymmetry / face_width
                        
                        total_asymmetry += normalized_asymmetry
                        valid_pairs += 1
                
                if valid_pairs > 0:
                    avg_asymmetry = total_asymmetry / valid_pairs
                    # Convert to symmetry score (0 = no symmetry, 1 = perfect symmetry)
                    symmetry = max(0.0, 1.0 - (avg_asymmetry * 5.0))  # Scale factor
                    return min(1.0, max(0.0, symmetry))
            
            return 0.5  # Default moderate symmetry
        except Exception:
            return 0.5
    
    def _get_face_width(self, landmarks_3d: List[Dict[str, float]]) -> float:
        """Calculate approximate face width from landmarks."""
        try:
            # Use ear to ear distance or eye to eye distance
            if len(landmarks_3d) > 454:  # right ear index
                left_ear = landmarks_3d[234]   # left ear
                right_ear = landmarks_3d[454]  # right ear
                
                dx = right_ear["x"] - left_ear["x"]
                dy = right_ear["y"] - left_ear["y"]
                dz = right_ear["z"] - left_ear["z"]
                
                return math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Fallback to eye distance
            if len(landmarks_3d) > 362:
                left_eye = landmarks_3d[33]   # left eye outer
                right_eye = landmarks_3d[362] # right eye outer
                
                dx = right_eye["x"] - left_eye["x"]
                dy = right_eye["y"] - left_eye["y"]
                dz = right_eye["z"] - left_eye["z"]
                
                return math.sqrt(dx*dx + dy*dy + dz*dz)
                
            return 0.1  # Default width
        except Exception:
            return 0.1
    
    def _calculate_expressivity(self, features: FacialFeatures) -> float:
        """Calculate overall facial expressivity score."""
        try:
            # Combine various expressive elements
            eye_expr = (features.eye_openness_left + features.eye_openness_right) / 2.0
            mouth_expr = features.mouth_width
            brow_expr = (features.brow_inner_up + 
                       (features.brow_down_left + features.brow_down_right) / 2.0 +
                       features.brow_outer_up_left + 
                       features.brow_outer_up_right) / 4.0
            movement_expr = min(1.0, features.facial_movement_velocity * 10.0)  # Scale factor
            
            # Weighted combination
            expressivity = (
                eye_expr * 0.3 +
                mouth_expr * 0.3 +
                brow_expr * 0.2 +
                movement_expr * 0.2
            )
            
            return min(1.0, max(0.0, expressivity))
        except Exception:
            return 0.0
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            # Eye features
            "eye_openness_left", "eye_openness_right", "eye_openness_avg",
            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate",
            
            # Mouth features
            "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry",
            # Phase 1 additions
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
            
            # Nose features
            "nose_sneer_left", "nose_sneer_right"
        ]
    
    def reset(self) -> None:
        """Reset internal state."""
        self._blink_state_history.clear()
        self._blink_timestamps.clear()
        self._last_landmarks = None
        self._last_timestamp = None