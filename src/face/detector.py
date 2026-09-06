"""
Face detection and landmark extraction using MediaPipe FaceLandmarker.

Detects faces, extracts 468 3D landmarks, blendshapes, and iris tracking.
"""

import time
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# Use the new MediaPipe Tasks API
from mediapipe.tasks.python.vision import (
    FaceLandmarker, 
    FaceLandmarkerOptions, 
    FaceLandmarkerResult, 
    RunningMode
)
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat

from core.logging import get_logger


@dataclass
class FaceDetectionResult:
    """Result of face detection and landmark extraction."""
    face_detected: bool
    detection_confidence: float
    bounding_box: Optional[Dict[str, int]]  # x, y, width, height in pixels
    landmarks_3d: Optional[List[Dict[str, float]]]  # 468 landmarks, normalized 0-1
    landmarks_2d: Optional[List[Dict[str, int]]]  # 468 landmarks, pixel coordinates
    blendshapes: Optional[Dict[str, float]]  # 52 blendshape coefficients
    iris: Optional[Dict[str, Dict[str, float]]]  # left/right iris centers
    head_pose: Optional[Dict[str, float]]  # pitch, yaw, roll in degrees
    processing_time_ms: float


class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe FaceLandmarker.
    
    Provides 468 3D landmarks, 52 blendshape coefficients, iris tracking, and head pose.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # MediaPipe settings
        self.detection_model = config.get("detection_model", "mediapipe")
        self.landmark_model = config.get("landmark_model", "mediapipe")
        self.refine_landmarks = config.get("refine_landmarks", True)
        self.min_detection_confidence = config.get("min_detection_confidence", 0.5)
        self.min_tracking_confidence = config.get("min_tracking_confidence", 0.5)
        
        # MediaPipe FaceLandmarker
        self._landmarker: Optional[FaceLandmarker] = None
        
        # For backwards compatibility with tests
        self._face_mesh = None
        
        # Blendshape names (52 MediaPipe blendshapes) - these are the standard FaceLandmarker blendshapes
        self._blendshape_names = [
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
        
        # Face found/lost tracking
        self._face_was_detected = False

    def _initialize(self) -> bool:
        """Initialize MediaPipe FaceLandmarker."""
        try:
            # Use the default model path or allow custom model
            model_path = self.config.get("model_path", "face_landmarker.task")
            
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                result_callback=self._result_callback
            )
            
            self._landmarker = FaceLandmarker.create_from_options(options)
            self._latest_result: Optional[FaceLandmarkerResult] = None
            self._result_ready = False
            return True
        except Exception as e:
            self.logger.error("face_detector_init_failed", error=str(e))
            return False

    def _result_callback(self, result: FaceLandmarkerResult, output_image: Image, timestamp_ms: int) -> None:
        """Callback for async face landmarker results."""
        self._latest_result = result
        self._result_ready = True

    def process_frame(self, frame: np.ndarray) -> FaceDetectionResult:
        """
        Process a frame to detect face and extract features.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            FaceDetectionResult with detection and feature data
        """
        start_time = time.perf_counter()
        
        # Initialize on first use
        if self._landmarker is None:
            if not self._initialize():
                return FaceDetectionResult(
                    face_detected=False,
                    detection_confidence=0.0,
                    bounding_box=None,
                    landmarks_3d=None,
                    landmarks_2d=None,
                    blendshapes=None,
                    iris=None,
                    head_pose=None,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Create MediaPipe Image
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Process frame (async)
        timestamp_ms = int(time.time() * 1000)
        self._result_ready = False
        self._landmarker.detect_async(mp_image, timestamp_ms)
        
        # Wait for result (with timeout)
        wait_start = time.perf_counter()
        while not self._result_ready:
            if (time.perf_counter() - wait_start) > 0.5:  # 500ms timeout
                self.logger.warning("face_detector_timeout")
                break
            time.sleep(0.001)
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        if self._latest_result is None or not self._latest_result.face_landmarks:
            # Face lost
            self._face_was_detected = False
            
            return FaceDetectionResult(
                face_detected=False,
                detection_confidence=0.0,
                bounding_box=None,
                landmarks_3d=None,
                landmarks_2d=None,
                blendshapes=None,
                iris=None,
                head_pose=None,
                processing_time_ms=processing_time_ms
            )
        
        # Face detected
        self._face_was_detected = True
        face_landmarks = self._latest_result.face_landmarks[0]
        
        # Get detection confidence from face landmarks
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
            landmarks_3d.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })
            landmarks_2d.append({
                "x": int(lm.x * w),
                "y": int(lm.y * h)
            })
        
        # Calculate bounding box from landmarks
        xs = [lm["x"] for lm in landmarks_2d]
        ys = [lm["y"] for lm in landmarks_2d]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bounding_box = {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }
        
        # Extract blendshapes (52 coefficients)
        blendshapes = {}
        if self._latest_result.face_blendshapes:
            for i, blendshape in enumerate(self._latest_result.face_blendshapes[0]):
                if i < len(self._blendshape_names):
                    blendshapes[self._blendshape_names[i]] = float(blendshape.score)
        
        # Extract iris centers (indices 468-477)
        iris = None
        try:
            # Iris landmarks: 468-477 (468-472 left iris, 473-477 right iris)
            # Center of left iris: 468, right iris: 473
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
        
        return FaceDetectionResult(
            face_detected=True,
            detection_confidence=detection_confidence,
            bounding_box=bounding_box,
            landmarks_3d=landmarks_3d,
            landmarks_2d=landmarks_2d,
            blendshapes=blendshapes,
            iris=iris,
            head_pose=head_pose,
            processing_time_ms=processing_time_ms
        )

    def _calculate_head_pose(self, face_landmarks, w: int, h: int) -> Dict[str, float]:
        """
        Calculate head pose (pitch, yaw, roll) from facial landmarks.
        """
        try:
            # Key landmark indices for head pose estimation
            NOSE_TIP = 1
            LEFT_EYE_OUTER = 33
            RIGHT_EYE_OUTER = 263
            CHIN = 152
            
            nose_tip = face_landmarks[NOSE_TIP]
            left_eye = face_landmarks[LEFT_EYE_OUTER]
            right_eye = face_landmarks[RIGHT_EYE_OUTER]
            chin = face_landmarks[CHIN]
            
            # Convert to pixel coordinates
            nose_2d = np.array([nose_tip.x * w, nose_tip.y * h])
            left_eye_2d = np.array([left_eye.x * w, left_eye.y * h])
            right_eye_2d = np.array([right_eye.x * w, right_eye.y * h])
            chin_2d = np.array([chin.x * w, chin.y * h])
            
            # Calculate yaw (left-right rotation)
            eye_center = (left_eye_2d + right_eye_2d) / 2
            nose_to_eye_center = nose_2d - eye_center
            eye_vector = right_eye_2d - left_eye_2d
            eye_distance = np.linalg.norm(eye_vector)
            
            if eye_distance > 0:
                # Yaw: based on nose offset from eye center
                yaw = np.degrees(np.arctan2(nose_to_eye_center[0], eye_distance * 0.5))
                
                # Pitch: based on nose to chin vertical relationship
                eye_to_chin = chin_2d - eye_center
                nose_to_eye = nose_2d - eye_center
                pitch = np.degrees(np.arctan2(nose_to_eye[1], np.linalg.norm(eye_to_chin) * 0.5))
                
                # Roll: based on eye line angle
                roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            else:
                yaw = pitch = roll = 0.0
            
            return {
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll)
            }
        except Exception as e:
            self.logger.debug("head_pose_calculation_failed", error=str(e))
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    def close(self) -> None:
        """Close MediaPipe resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            self._face_mesh = None
            self.logger.info("face_detector_closed")