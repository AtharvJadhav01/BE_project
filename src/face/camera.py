"""
Face camera capture using OpenCV.

Captures frames from webcam for face processing.
"""

import threading
import time
import cv2
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass

from core.events import Event, EventType, ModalityType, create_face_event
from core.logging import get_logger
from core.timestamps import TimestampGenerator


@dataclass
class CameraFrame:
    """Raw camera frame data."""
    frame: np.ndarray
    timestamp: float  # monotonic time
    frame_index: int


class FaceCamera:
    """
    Webcam capture using OpenCV.
    
    Captures frames at configured FPS for face processing.
    """

    def __init__(
        self,
        session_id: str,
        timestamp_generator: TimestampGenerator,
        event_callback: Callable[[Event], None],
        config: Dict[str, Any]
    ):
        self.session_id = session_id
        self.timestamp_generator = timestamp_generator
        self.event_callback = event_callback
        self.config = config
        self.logger = get_logger(__name__)
        
        # Camera settings
        self.fps = config.get("fps", 15)
        self.resolution = config.get("resolution", [640, 480])
        self.device = config.get("device", 0)
        
        # Calculate frame interval
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 0
        
        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        
        # Worker thread for capturing frames
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_queue: Optional["queue.Queue"] = None
        
        # Frame tracking
        self._frame_index = 0
        self._last_frame_time = 0.0
        
        # Processing callback
        self._process_callback: Optional[Callable[[CameraFrame], None]] = None

    def set_process_callback(self, callback: Callable[[CameraFrame], None]) -> None:
        """Set callback for processing captured frames."""
        self._process_callback = callback

    def _capture_loop(self) -> None:
        """Background thread to capture frames at target FPS."""
        while self._running:
            loop_start = time.perf_counter()
            
            with self._lock:
                if not self._running or self._paused:
                    time.sleep(0.01)
                    continue
                
                # Capture frame
                ret, frame = self._cap.read()
                if not ret:
                    self.logger.warning("camera_frame_capture_failed")
                    time.sleep(0.01)
                    continue
                
                current_time = self.timestamp_generator.now_monotonic()
                self._frame_index += 1
                
                # Create frame data
                frame_data = CameraFrame(
                    frame=frame.copy(),
                    timestamp=current_time,
                    frame_index=self._frame_index
                )
                
                # Emit face_found event if this is the first frame or face was lost
                # (We'll let the processor handle face detection events)
                
                # Forward to processor
                if self._process_callback:
                    try:
                        self._process_callback(frame_data)
                    except Exception as e:
                        self.logger.error("face_process_callback_error", error=str(e))
            
            # Maintain target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> bool:
        """Start the camera capture."""
        if self._running:
            return True
        
        try:
            # Open camera
            self._cap = cv2.VideoCapture(self.device)
            if not self._cap.isOpened():
                self.logger.error("camera_open_failed", device=self.device)
                return False
            
            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info("camera_started", 
                           session_id=self.session_id,
                           device=self.device,
                           requested_resolution=self.resolution,
                           actual_resolution=[actual_width, actual_height],
                           requested_fps=self.fps,
                           actual_fps=actual_fps)
            
            self._running = True
            self._paused = False
            self._frame_index = 0
            
            # Start capture thread
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error("camera_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the camera capture."""
        if not self._running:
            return True
        
        try:
            self._running = False
            
            if self._capture_thread:
                self._capture_thread.join(timeout=2.0)
                self._capture_thread = None
            
            if self._cap:
                self._cap.release()
                self._cap = None
            
            self.logger.info("camera_stopped", session_id=self.session_id)
            return True
            
        except Exception as e:
            self.logger.error("camera_stop_failed", error=str(e))
            return False

    def pause(self) -> bool:
        """Pause the camera capture."""
        if not self._running or self._paused:
            return False
        
        with self._lock:
            self._paused = True
            self.logger.info("camera_paused", session_id=self.session_id)
            return True

    def resume(self) -> bool:
        """Resume the camera capture."""
        if not self._running or not self._paused:
            return False
        
        with self._lock:
            self._paused = False
            self.logger.info("camera_resumed", session_id=self.session_id)
            return True

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    @staticmethod
    def check_camera_availability(device: int = 0) -> bool:
        """Check if a camera device is available."""
        try:
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                cap.release()
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def get_available_cameras(max_devices: int = 10) -> list:
        """Get list of available camera devices."""
        cameras = []
        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cameras.append({
                        "index": i,
                        "width": width,
                        "height": height,
                    })
                    cap.release()
            except Exception:
                pass
        return cameras