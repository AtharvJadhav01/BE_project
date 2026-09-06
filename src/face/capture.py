"""
Face capture modality implementation.

Implements the FaceModalityInterface using OpenCV + MediaPipe for face capture.
"""

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from core.events import Event, EventType, ModalityType, create_face_event
from core.interfaces import FaceModalityInterface, ModalityStatus
from core.logging import get_logger
from core.timestamps import TimestampGenerator

from .camera import FaceCamera, CameraFrame
from .detector import FaceDetector, FaceDetectionResult
from .processor import FaceProcessor, ProcessedFaceObservation
from .storage import FaceStorage


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
        
        # Session state
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._timestamp_generator: Optional[TimestampGenerator] = None
        self._unified_event_callback: Optional[callable] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Frame tracking
        self._frame_count = 0

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
            self._frame_count = 0
            
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
                
                # Start components in order
                if not self._storage.start():
                    raise RuntimeError("Failed to start face storage")
                
                if not self._processor.start():
                    raise RuntimeError("Failed to start face processor")
                
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
        self._frame_count = 0

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
        
        return stats