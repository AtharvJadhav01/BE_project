"""
Face processor for handling face detection results.

Processes face detection results and forwards to storage/unified pipeline.
"""

import threading
import queue
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from queue import Empty

from core.logging import get_logger
from .detector import FaceDetectionResult


@dataclass
class ProcessedFaceObservation:
    """Processed face observation ready for storage."""
    observation_id: str
    frame_index: int
    timestamp_monotonic: float
    timestamp_wallclock: str
    face_detected: bool
    detection_confidence: float
    bounding_box: Optional[Dict[str, int]]
    landmarks_3d: Optional[List[Dict[str, float]]]
    landmarks_2d: Optional[List[Dict[str, int]]]
    blendshapes: Optional[Dict[str, float]]
    iris: Optional[Dict[str, Dict[str, float]]]
    head_pose: Optional[Dict[str, float]]
    processing_time_ms: float


class FaceProcessor:
    """
    Processes face detection results from the detector.
    
    Handles validation, enrichment, and forwarding to storage.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Event queue for async processing
        self._observation_queue: queue.Queue = queue.Queue(maxsize=100)
        self._processing = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._storage_callback: Optional[Callable] = None
        self._unified_event_callback: Optional[Callable] = None
        
        # Statistics
        self._observations_processed = 0
        self._observations_dropped = 0
        
        # Face state tracking
        self._face_was_detected = False

    def set_storage_callback(self, callback: Callable) -> None:
        """Set callback for storing processed observations."""
        self._storage_callback = callback

    def set_unified_event_callback(self, callback: Callable) -> None:
        """Set callback for unified event pipeline."""
        self._unified_event_callback = callback

    def start(self) -> bool:
        """Start the processor."""
        if self._processing:
            return True
        
        self._processing = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        self.logger.info("face_processor_started")
        return True

    def stop(self) -> bool:
        """Stop the processor."""
        if not self._processing:
            return True
        
        self._processing = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self.logger.info("face_processor_stopped", 
                        processed=self._observations_processed, 
                        dropped=self._observations_dropped)
        return True

    def process_observation(self, result: FaceDetectionResult, frame_index: int, 
                          timestamp_monotonic: float, timestamp_wallclock: str) -> bool:
        """
        Queue a face observation for processing.
        
        Args:
            result: Face detection result
            frame_index: Sequential frame index
            timestamp_monotonic: Monotonic timestamp
            timestamp_wallclock: Wallclock timestamp (ISO 8601)
            
        Returns:
            True if queued successfully, False if queue full
        """
        if not self._processing:
            return False
        
        try:
            observation = ProcessedFaceObservation(
                observation_id=f"face_{frame_index:06d}",
                frame_index=frame_index,
                timestamp_monotonic=timestamp_monotonic,
                timestamp_wallclock=timestamp_wallclock,
                face_detected=result.face_detected,
                detection_confidence=result.detection_confidence,
                bounding_box=result.bounding_box,
                landmarks_3d=result.landmarks_3d,
                landmarks_2d=result.landmarks_2d,
                blendshapes=result.blendshapes,
                iris=result.iris,
                head_pose=result.head_pose,
                processing_time_ms=result.processing_time_ms,
            )
            self._observation_queue.put_nowait(observation)
            return True
        except queue.Full:
            self._observations_dropped += 1
            self.logger.warning("face_processor_queue_full", dropped=self._observations_dropped)
            return False
        except Exception as e:
            self.logger.error("face_processor_enqueue_failed", error=str(e))
            return False

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._processing:
            try:
                observation = self._observation_queue.get(timeout=0.1)
                self._process_single_observation(observation)
            except Empty:
                continue
            except Exception as e:
                self.logger.error("face_processor_error", error=str(e))

    def _process_single_observation(self, observation: ProcessedFaceObservation) -> None:
        """Process a single face observation."""
        try:
            # Track face state changes
            face_detected = observation.face_detected
            
            if face_detected and not self._face_was_detected:
                # Face found event
                self._emit_face_event("face_found", observation)
            elif not face_detected and self._face_was_detected:
                # Face lost event
                self._emit_face_event("face_lost", observation)
            
            self._face_was_detected = face_detected
            
            # Always emit face_observation event
            self._emit_face_event("face_observation", observation)
            
            # Forward to storage
            if self._storage_callback:
                self._storage_callback(observation)
            
            self._observations_processed += 1
            
        except Exception as e:
            self.logger.error("face_observation_processing_failed", 
                            observation_id=observation.observation_id, 
                            error=str(e))

    def _emit_face_event(self, event_type: str, observation: ProcessedFaceObservation) -> None:
        """Emit face event to unified pipeline."""
        if self._unified_event_callback:
            from core.events import create_face_event, EventType
            event = create_face_event(
                session_id="",  # Will be filled by caller
                event_type=EventType(event_type),
                data=observation.__dict__,
            )
            self._unified_event_callback(event)

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "observations_processed": self._observations_processed,
            "observations_dropped": self._observations_dropped,
            "queue_size": self._observation_queue.qsize(),
        }