"""
Face capture modality implementation.

Implements the FaceModalityInterface using OpenCV + MediaPipe for face capture.
"""

import math
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.events import Event, EventType, ModalityType, create_face_event
from core.interfaces import FaceModalityInterface, ModalityStatus
from core.logging import get_logger
from core.timestamps import TimestampGenerator

from .camera import FaceCamera, CameraFrame
from .detector import FaceDetector, FaceDetectionResult
from .processor import FaceProcessor, ProcessedFaceObservation
from .storage import FaceStorage

# Analysis imports
from analysis.face.features import FaceFeatureExtractor
from analysis.face.state import FaceStateEstimator
from analysis.face.baseline import FaceBaselineManager
from analysis.face.emotion_classifier import FaceEmotionClassifier, EmotionClassifierError
from analysis.face.state import EmotionSmoother
from analysis.temporal import TemporalWindow
from analysis.results import FeatureVector, StateEstimate, AnalysisResult
import orjson


class FaceCapture(FaceModalityInterface):
    """
    Face capture modality.
    
    Captures webcam frames, detects faces, extracts landmarks/expressions,
    and stores observations as JSONL with metadata.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Components
        self._camera: Optional[FaceCamera] = None
        self._detector: Optional[FaceDetector] = None
        self._processor: Optional[FaceProcessor] = None
        self._storage: Optional[FaceStorage] = None
        
        # Analysis components
        self._feature_extractor: Optional[FaceFeatureExtractor] = None
        self._baseline_manager: Optional[FaceBaselineManager] = None
        self._temporal_windows: Optional[Dict[str, TemporalWindow]] = None
        self._state_estimator: Optional[FaceStateEstimator] = None
        self._emotion_classifier: Optional[FaceEmotionClassifier] = None
        self._analysis_storage: Optional[Any] = None  # Will be analysis storage writer
        
        # Session state
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._timestamp_generator: Optional[TimestampGenerator] = None
        self._unified_event_callback: Optional[callable] = None
        
        # Analysis state
        self._last_analysis_result: Optional[Dict[str, Any]] = None
        self._analysis_frame_count = 0
        self._last_analysis_timestamp: Optional[float] = None
        
        # Emotion estimation state
        self._emotion_enabled: bool = False
        self._last_emotion_result: Optional[Dict[str, Any]] = None
        
        # Thread safety
        self._lock = threading.Lock()

    def set_unified_event_callback(self, callback: callable) -> None:
        """Set callback for unified event pipeline."""
        self._unified_event_callback = callback

    def _on_camera_frame(self, frame: CameraFrame) -> None:
        """Callback for camera frames - runs face detection."""
        if not self._detector or not self._processor:
            return
        
        # Run face detection
        result = self._detector.process_frame(frame.frame)
        
        # Get timestamps
        timestamp_monotonic = frame.timestamp
        timestamp_wallclock = self._timestamp_generator.now_wallclock() if self._timestamp_generator else ""
        
        # Process observation
        self._processor.process_observation(
            result=result,
            frame_index=frame.frame_index,
            timestamp_monotonic=timestamp_monotonic,
            timestamp_wallclock=timestamp_wallclock
        )

    def _on_storage_write(self, observation) -> None:
        """Callback for writing observations to storage."""
        if self._storage:
            self._storage.write_observation(observation)

    def _on_face_event(self, event: Event) -> None:
        """Callback for face events to unified pipeline."""
        if self._unified_event_callback:
            # Inject session_id
            event.data["session_id"] = self._session_id
            self._unified_event_callback(event)

    def start(self, session_id: str, session_dir: Optional[Path] = None) -> bool:
        """
        Start face capture for a session.
        
        Args:
            session_id: Unique session identifier
            session_dir: Optional session directory path
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._status == ModalityStatus.RUNNING:
                self.logger.warning("face_already_running", session_id=session_id)
                return True
            
            self._session_id = session_id
            
            # Use provided session_dir or construct from config
            if session_dir is not None:
                self._session_dir = session_dir
            else:
                self._session_dir = Path(self.config.get("storage", {}).get("output_directory", "data/sessions")) / session_id
            
            self._timestamp_generator = TimestampGenerator()
            self._analysis_frame_count = 0
            self._last_analysis_timestamp = None
            
            try:
                # Check camera availability
                if not FaceCamera.check_camera_availability(self.config.get("device", 0)):
                    self.logger.error("face_no_camera_available")
                    self._set_error(RuntimeError("No camera available"), "No camera available")
                    return False
                
                # Initialize components
                self._detector = FaceDetector(self.config.get("face", {}))
                
                self._processor = FaceProcessor(self.config.get("face", {}))
                self._processor.set_storage_callback(self._on_storage_write)
                self._processor.set_unified_event_callback(self._on_face_event)
                
                self._storage = FaceStorage(self._session_dir, self.config, self._timestamp_generator)
                
                self._camera = FaceCamera(
                    session_id=session_id,
                    timestamp_generator=self._timestamp_generator,
                    event_callback=self._on_face_event,  # This won't be used directly for frames
                    config=self.config.get("camera", {})
                )
                self._camera.set_process_callback(self._on_camera_frame)
                
                # Initialize analysis components
                self._initialize_analysis_components()
                
                # Start components in order
                if not self._storage.start():
                    raise RuntimeError("Failed to start face storage")
                
                if not self._processor.start():
                    raise RuntimeError("Failed to start face processor")
                
                if not self._analysis_start():
                    raise RuntimeError("Failed to start face analysis")
                
                if not self._camera.start():
                    raise RuntimeError("Failed to start face camera")
                
                self._set_status(ModalityStatus.RUNNING)
                self.logger.info("face_capture_started", 
                               session_id=session_id,
                               fps=self.config.get("camera", {}).get("fps", 15),
                               resolution=self.config.get("camera", {}).get("resolution", [640, 480]))
                return True
                
            except Exception as e:
                self._set_error(e, f"Failed to start face capture: {e}")
                self._cleanup()
                return False

    def stop(self) -> bool:
        """
        Stop face capture.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if self._status == ModalityStatus.STOPPED:
                return True
            
            self._set_status(ModalityStatus.STOPPING)
            
            success = True
            
            # Stop in reverse order
            if self._camera:
                if not self._camera.stop():
                    success = False
            
            if self._processor:
                if not self._processor.stop():
                    success = False
            
            # Stop analysis components
            if not self._analysis_stop():
                success = False
            
            if self._storage:
                if not self._storage.stop():
                    success = False
            
            if self._detector:
                self._detector.close()
            
            self._cleanup()
            
            if success:
                self._set_status(ModalityStatus.STOPPED)
                self.logger.info("face_capture_stopped", session_id=self._session_id)
            else:
                self._set_status(ModalityStatus.ERROR)
                self.logger.error("face_capture_stop_failed", session_id=self._session_id)
            
            return success

    def pause(self) -> bool:
        """
        Pause face capture.
        
        Returns:
            True if paused successfully, False otherwise
        """
        with self._lock:
            if self._status != ModalityStatus.RUNNING:
                return False
            
            if self._camera:
                result = self._camera.pause()
                if result:
                    self._set_status(ModalityStatus.PAUSED)
                    self.logger.info("face_capture_paused", session_id=self._session_id)
                return result
            return False

    def resume(self) -> bool:
        """
        Resume face capture.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        with self._lock:
            if self._status != ModalityStatus.PAUSED:
                return False
            
            if self._camera:
                result = self._camera.resume()
                if result:
                    self._set_status(ModalityStatus.RUNNING)
                    self.logger.info("face_capture_resumed", session_id=self._session_id)
                return result
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._camera = None
        self._detector = None
        self._processor = None
        self._storage = None
        self._session_id = None
        self._session_dir = None
        self._timestamp_generator = None
        
        # Clear analysis components
        self._feature_extractor = None
        self._baseline_manager = None
        self._temporal_windows = None
        self._state_estimator = None
        self._emotion_classifier = None
        self._last_analysis_result = None
        self._last_emotion_result = None
        self._emotion_enabled = False

    def set_emotion_enabled(self, enabled: bool) -> None:
        """
        Enable or disable emotion estimation at runtime.
        
        Args:
            enabled: True to enable, False to disable
        """
        with self._lock:
            self._emotion_enabled = enabled
            if not enabled and self._emotion_classifier:
                self._emotion_classifier.reset()
            self.logger.info("emotion_estimation_toggled", enabled=enabled)

    def is_emotion_enabled(self) -> bool:
        """Check if emotion estimation is currently enabled."""
        return self._emotion_enabled

    def reset_emotion_classifier(self) -> None:
        """Reset the emotion classifier state."""
        with self._lock:
            if self._emotion_classifier:
                self._emotion_classifier.reset()
            self._last_emotion_result = None
            self.logger.info("emotion_classifier_reset")

    # Analysis methods
    def _initialize_analysis_components(self) -> None:
        """Initialize face analysis components."""
        try:
            # Get face analysis config
            face_config = self.config.get("analysis", {}).get("face", {})
            analysis_config = self.config.get("analysis", {})
            
            # Check if emotion estimation is enabled
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
            )
            
            # Initialize feature extractor
            self._feature_extractor = FaceFeatureExtractor(
                "face_features", 
                face_config
            )
            
            # Initialize baseline manager
            self._baseline_manager = FaceBaselineManager(
                warmup_duration_sec=analysis_config.get("baseline_warmup_sec", 10.0),
                min_samples=analysis_config.get("baseline_min_samples", 50),
                ema_alpha=analysis_config.get("ema_alpha", 0.1),
                min_std=analysis_config.get("min_std", 1e-6)
            )
            
            # Initialize temporal windows for key facial features
            # We'll create windows for the most important features
            key_features = [
                "eye_openness_left", "eye_openness_right", "eye_openness_avg",
                "mouth_openness", "mouth_width", 
                "brow_inner_up", "brow_down_left", "brow_down_right",
                "head_pitch", "head_yaw", "head_roll",
                "facial_movement_velocity", "facial_symmetry",
                "overall_expressivity"
            ]
            
            window_size = analysis_config.get("temporal_window_sec", 3.0)
            self._temporal_windows = {
                feature: TemporalWindow(window_duration_sec=window_size)
                for feature in key_features
            }
            
            # Initialize state estimator
            self._state_estimator = FaceStateEstimator(
                "face_state", 
                face_config
            )
            
            # Initialize emotion classifier if enabled
            if self._emotion_enabled:
                try:
                    self._emotion_classifier = FaceEmotionClassifier(config=face_config)
                    if not self._emotion_classifier.is_ready():
                        self.logger.warning(
                            "emotion_classifier_not_ready",
                            error=self._emotion_classifier.get_load_error()
                        )
                        self._emotion_classifier = None
                    else:
                        self.logger.info("emotion_classifier_initialized")
                except EmotionClassifierError as e:
                    self.logger.warning("emotion_classifier_init_failed", error=str(e))
                    self._emotion_classifier = None
            
            # Initialize analysis storage
            self._setup_analysis_storage()
            
            self.logger.info("face_analysis_components_initialized", 
                           emotion_enabled=self._emotion_enabled,
                           emotion_classifier_ready=self._emotion_classifier is not None)
            
        except Exception as e:
            self.logger.error("face_analysis_init_failed", error=str(e))
            raise

    def _setup_analysis_storage(self) -> None:
        """Set up analysis storage for facial features, states, and expressions."""
        try:
            # Create analysis directory
            analysis_dir = self._session_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # We'll implement simple file writing for now
            # In a full implementation, this would use async writing like FaceStorage
            self._analysis_features_path = analysis_dir / "face_features.jsonl"
            self._analysis_states_path = analysis_dir / "face_states.jsonl"
            self._analysis_expressions_path = analysis_dir / "face_expressions.jsonl"
            
            # Open files for appending
            self._analysis_features_file = open(self._analysis_features_path, "ab")
            self._analysis_states_file = open(self._analysis_states_path, "ab")
            self._analysis_expressions_file = open(self._analysis_expressions_path, "ab")
            
            self.logger.info("face_analysis_storage_setup", 
                           features_path=str(self._analysis_features_path),
                           states_path=str(self._analysis_states_path),
                           expressions_path=str(self._analysis_expressions_path))
                           
        except Exception as e:
            self.logger.error("face_analysis_storage_setup_failed", error=str(e))
            raise

    def _analysis_start(self) -> bool:
        """Start analysis components."""
        try:
            # Files are already opened in _setup_analysis_storage
            # Just mark as ready
            self._analysis_ready = True
            self.logger.info("face_analysis_started")
            return True
        except Exception as e:
            self.logger.error("face_analysis_start_failed", error=str(e))
            return False

    def _on_camera_frame(self, frame: CameraFrame) -> None:
        """Callback for camera frames - runs face detection and analysis."""
        if not self._detector or not self._processor:
            return
        
        # Run face detection
        result = self._detector.process_frame(frame.frame)
        
        # Get timestamps
        timestamp_monotonic = frame.timestamp
        timestamp_wallclock = self._timestamp_generator.now_wallclock() if self._timestamp_generator else ""
        
        # Run face analysis if enabled and we have a face
        analysis_result = None
        if (self._feature_extractor is not None and 
            self._baseline_manager is not None and
            self._state_estimator is not None and
            result.face_detected):
            
            analysis_result = self._run_face_analysis(
                result, timestamp_monotonic, timestamp_wallclock, frame.frame_index
            )
        
        # Process observation (original detection data)
        self._processor.process_observation(
            result=result,
            frame_index=frame.frame_index,
            timestamp_monotonic=timestamp_monotonic,
            timestamp_wallclock=timestamp_wallclock
        )
        
        # Store analysis results if we have them
        if analysis_result is not None:
            self._store_analysis_results(analysis_result, frame.frame_index)

    def _run_face_analysis(self, detection_result: FaceDetectionResult,
                         timestamp_monotonic: float,
                         timestamp_wallclock: str,
                         frame_index: int) -> Optional[Dict[str, Any]]:
        """
        Run face analysis pipeline on detection results.
        
        Args:
            detection_result: Result from face detector
            timestamp_monotonic: Monotonic timestamp
            timestamp_wallclock: Wallclock timestamp
            frame_index: Frame index for this analysis
            
        Returns:
            Dictionary containing analysis results or None if analysis failed
        """
        try:
            # Extract features from detection result
            raw_features = self._extract_raw_features_from_detection(detection_result, timestamp_monotonic, timestamp_wallclock)
            
            if not raw_features:
                return None
            
            # Update temporal windows
            self._update_temporal_windows(raw_features, timestamp_monotonic)
            
            # Get temporal window statistics
            window_stats = self._get_temporal_window_stats()
            
            # Update baseline with window statistics
            self._update_baseline(window_stats, timestamp_monotonic)
            
            # Normalize features using baseline
            normalized_features = self._normalize_features(window_stats)
            
            # Estimate facial state (rule-based)
            state_estimate = self._estimate_facial_state(normalized_features)
            
            # Estimate expression (ML-based) if enabled and face detected
            expression_estimate = None
            if (self._emotion_enabled and 
                self._emotion_classifier is not None and 
                detection_result.face_detected):
                
                # Add prediction to smoother
                try:
                    raw_probs = self._emotion_classifier.get_metadata().get("target_names", EMOTION_LABELS)
                    # Get the raw prediction first
                    result = self._emotion_classifier.predict(raw_features)
                    probas = result["probabilities"]
                    # Convert to numpy array in correct order
                    probs_array = np.array([probas[label] for label in raw_probs], dtype=np.float64)
                    confidence = result["confidence"]
                    self._emotion_smoother.add_prediction(probs_array, confidence)
                    
                    # Get smoothed result
                    smoothed_expr, smoothed_conf = self._emotion_smoother.get_smoothed()
                    
                    if smoothed_expr is not None:
                        expression_estimate = {
                            "expression": smoothed_expr,
                            "confidence": smoothed_conf,
                            "probabilities": result["probabilities"],
                            "timestamp": timestamp_monotonic,
                            "wallclock": timestamp_wallclock,
                        }
                    else:
                        # No valid smoothed prediction - check if we should show unavailable
                        if not detection_result.face_detected:
                            expression_estimate = {
                                "expression": "unavailable",
                                "confidence": 0.0,
                                "probabilities": {},
                                "timestamp": timestamp_monotonic,
                                "wallclock": timestamp_wallclock,
                                "reason": "no_face_detected"
                            }
                        else:
                            # Classifier not confident enough, show unavailable
                            expression_estimate = {
                                "expression": "unavailable",
                                "confidence": 0.0,
                                "probabilities": {},
                                "timestamp": timestamp_monotonic,
                                "wallclock": timestamp_wallclock,
                                "reason": "low_confidence"
                            }
                except Exception as e:
                    self.logger.warning("emotion_smoothing_failed", error=str(e))
                    # Fall back to raw prediction without smoothing
                    expression_estimate = self._emotion_classifier.predict(raw_features)
                try:
                    expression_estimate = self._emotion_classifier.predict(
                        raw_features,  # Use raw features for ML classifier (matching training)
                        timestamp_monotonic
                    )
                    self._last_emotion_result = expression_estimate
                except EmotionClassifierError as e:
                    self.logger.warning("emotion_prediction_failed", error=str(e))
                    expression_estimate = {
                        "expression": "unavailable",
                        "confidence": 0.0,
                        "probabilities": {},
                        "timestamp": timestamp_monotonic,
                        "wallclock": timestamp_wallclock,
                        "error": str(e)
                    }
            elif not detection_result.face_detected:
                # No face detected - expression unavailable (not guessing)
                expression_estimate = {
                    "expression": "unavailable",
                    "confidence": 0.0,
                    "probabilities": {},
                    "timestamp": timestamp_monotonic,
                    "wallclock": timestamp_wallclock,
                    "reason": "no_face_detected"
                }
            
            # Create analysis result
            analysis_result = {
                "timestamp_monotonic": timestamp_monotonic,
                "timestamp_wallclock": timestamp_wallclock,
                "frame_index": frame_index,
                "raw_features": raw_features,
                "window_stats": window_stats,
                "normalized_features": normalized_features,
                "state_estimate": state_estimate,
                "expression_estimate": expression_estimate,
                "baseline_ready": self._is_baseline_ready(),
                "analysis_timestamp": time.monotonic()
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("face_analysis_pipeline_failed", error=str(e))
            return None

    def _extract_raw_features_from_detection(self, detection_result: FaceDetectionResult, timestamp_monotonic: float, timestamp_wallclock: str) -> Dict[str, float]:
        """Extract raw facial features from FaceDetectionResult."""
        try:
            # Convert detection result to format expected by feature extractor
            raw_data = {
                "face_detected": detection_result.face_detected,
                "landmarks_3d": detection_result.landmarks_3d,
                "blendshapes": detection_result.blendshapes,
                "head_pose": detection_result.head_pose,
                "iris": detection_result.iris
            }
            
            # Use feature extractor to get features
            if self._feature_extractor:
                # Pass the current timestamps for proper feature extraction
                features = self._feature_extractor.extract(
                    raw_data, 
                    timestamp=timestamp_monotonic,
                    wallclock=timestamp_wallclock
                )
                
                if features is not None:
                    return features.features
            
            return {}
        except Exception as e:
            self.logger.error("face_feature_extraction_failed", error=str(e))
            return {}

    def _update_temporal_windows(self, features: Dict[str, float], timestamp: float) -> None:
        """Update temporal windows with feature values."""
        if self._temporal_windows is None:
            return
            
        for feature_name, value in features.items():
            if feature_name in self._temporal_windows:
                self._temporal_windows[feature_name].add(value, timestamp)

    def _get_temporal_window_stats(self) -> Dict[str, float]:
        """Get statistics from all temporal windows."""
        stats = {}
        if self._temporal_windows is not None:
            for feature_name, window in self._temporal_windows.items():
                window_stat = window.get_stats()
                if window_stat.count > 0:  # Only include if we have data
                    stats[feature_name] = window_stat.mean
        return stats

    def _update_baseline(self, features: Dict[str, float], timestamp: float) -> None:
        """Update baseline with feature values."""
        if self._baseline_manager is not None:
            for feature_name, value in features.items():
                self._baseline_manager.update(feature_name, value, timestamp)

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using baseline manager."""
        if self._baseline_manager is None:
            return features
            
        normalized = {}
        for feature_name, value in features.items():
            normalized_value = self._baseline_manager.normalize(feature_name, value)
            if normalized_value is not None:
                normalized[feature_name] = normalized_value
            else:
                # If baseline not ready, use raw value (will be handled in state estimation)
                normalized[feature_name] = value
                
        return normalized

    def _estimate_facial_state(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Estimate facial state from normalized features."""
        if self._state_estimator is None or self._baseline_manager is None:
            return None
            
        try:
            # Check if baseline is ready
            if not self._is_baseline_ready():
                return {
                    "state": "uncertain",
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "contributing_features": {}
                }
            
            # Get state estimate from estimator
            state_result = self._state_estimator.estimate(features, self._baseline_manager)
            
            if state_result is not None:
                return {
                    "state": state_result.state,
                    "confidence": state_result.confidence,
                    "uncertainty": state_result.uncertainty,
                    "contributing_features": state_result.contributing_features
                }
            
            return None
        except Exception as e:
            self.logger.error("face_state_estimation_failed", error=str(e))
            return None

    def _is_baseline_ready(self) -> bool:
        """Check if face baseline is ready."""
        if self._baseline_manager is None:
            return False
        return self._baseline_manager.is_face_baseline_ready()

    def _store_analysis_results(self, analysis_result: Dict[str, Any], frame_index: int) -> None:
        """Store analysis results to analysis files."""
        try:
            # Store facial features
            feature_line = {
                "timestamp_monotonic": analysis_result["timestamp_monotonic"],
                "timestamp_wallclock": analysis_result["timestamp_wallclock"],
                "frame_index": frame_index,
                "features": analysis_result["normalized_features"],
                "raw_features": analysis_result["raw_features"],
                "window_stats": analysis_result["window_stats"],
                "baseline_ready": analysis_result["baseline_ready"]
            }
            
            # Write to features file
            import orjson
            line = orjson.dumps(feature_line) + b"\n"
            self._analysis_features_file.write(line)
            self._analysis_features_file.flush()
            
            # Store facial state
            state_line = {
                "timestamp_monotonic": analysis_result["timestamp_monotonic"],
                "timestamp_wallclock": analysis_result["timestamp_wallclock"],
                "frame_index": frame_index,
                "state": analysis_result["state_estimate"]["state"] if analysis_result["state_estimate"] else "unknown",
                "confidence": analysis_result["state_estimate"]["confidence"] if analysis_result["state_estimate"] else 0.0,
                "uncertainty": analysis_result["state_estimate"]["uncertainty"] if analysis_result["state_estimate"] else 1.0,
                "contributing_features": analysis_result["state_estimate"]["contributing_features"] if analysis_result["state_estimate"] else {},
                "baseline_ready": analysis_result["baseline_ready"]
            }
            
            # Write to states file
            line = orjson.dumps(state_line) + b"\n"
            self._analysis_states_file.write(line)
            self._analysis_states_file.flush()
            
            # Store expression estimate if available
            expr_estimate = analysis_result.get("expression_estimate")
            if expr_estimate:
                expr_line = {
                    "timestamp_monotonic": analysis_result["timestamp_monotonic"],
                    "timestamp_wallclock": analysis_result["timestamp_wallclock"],
                    "frame_index": frame_index,
                    "expression": expr_estimate.get("expression", "unavailable"),
                    "confidence": expr_estimate.get("confidence", 0.0),
                    "probabilities": expr_estimate.get("probabilities", {}),
                    "baseline_ready": analysis_result["baseline_ready"]
                }
                if hasattr(self, '_analysis_expressions_file') and self._analysis_expressions_file:
                    line = orjson.dumps(expr_line) + b"\n"
                    self._analysis_expressions_file.write(line)
                    self._analysis_expressions_file.flush()
            
            self._analysis_frame_count += 1
            self._last_analysis_timestamp = analysis_result["timestamp_monotonic"]
            # Store last analysis result for UI polling
            self._last_analysis_result = analysis_result
            
        except Exception as e:
            self.logger.error("face_analysis_storage_failed", error=str(e))

    def _analysis_stop(self) -> bool:
        """Stop analysis components and close files."""
        try:
            # Close analysis files
            if hasattr(self, '_analysis_features_file') and self._analysis_features_file:
                self._analysis_features_file.close()
            if hasattr(self, '_analysis_states_file') and self._analysis_states_file:
                self._analysis_states_file.close()
            if hasattr(self, '_analysis_expressions_file') and self._analysis_expressions_file:
                self._analysis_expressions_file.close()
            
            self.logger.info("face_analysis_stopped", 
                           frames_processed=self._analysis_frame_count)
            return True
        except Exception as e:
            self.logger.error("face_analysis_stop_failed", error=str(e))
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        stats = {
            "status": self._status.value,
            "session_id": self._session_id,
        }
        
        if self._camera:
            stats["camera"] = {
                "running": self._camera.is_running,
                "paused": self._camera.is_paused,
            }
        
        if self._processor:
            stats["processor"] = self._processor.get_stats()
        
        if self._storage:
            stats["storage"] = self._storage.get_stats()
        
        # Add analysis stats
        stats["analysis"] = {
            "frames_processed": getattr(self, '_analysis_frame_count', 0),
            "last_analysis_timestamp": getattr(self, '_last_analysis_timestamp', None),
            "features_file": getattr(self, '_analysis_features_path', None),
            "states_file": getattr(self, '_analysis_states_path', None)
        }
        
        return stats