"""
Face storage for writing facial observations to JSONL.

Writes facial observations to facial_features.jsonl.
"""

import threading
import orjson
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from queue import Queue, Empty

from core.logging import get_logger
from core.timestamps import TimestampGenerator
from .processor import ProcessedFaceObservation


class FaceStorage:
    """
    Stores facial observations to JSONL file.
    
    Uses async writing with a background thread for performance.
    """

    def __init__(self, session_dir: Path, config: Dict[str, Any],
                 timestamp_generator: Optional[TimestampGenerator] = None):
        self.session_dir = session_dir
        self.config = config
        self.logger = get_logger(__name__)
        
        # Timestamp generator for wallclock conversion
        self._timestamp_generator = timestamp_generator
        
        # File path
        self._file_path = session_dir / "facial_features.jsonl"
        
        # Async writing
        self._write_queue: Queue = Queue()
        self._writing = False
        self._writer_thread: Optional[threading.Thread] = None
        self._file_handle = None
        
        # Statistics
        self._observations_written = 0
        self._write_errors = 0

    def start(self) -> bool:
        """Start the storage writer."""
        if self._writing:
            return True
        
        try:
            # Ensure directory exists
            self.session_dir.mkdir(parents=True, exist_ok=True)
            
            # Open file for appending
            self._file_handle = open(self._file_path, "ab")
            
            self._writing = True
            self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self._writer_thread.start()
            
            self.logger.info("face_storage_started", path=str(self._file_path))
            return True
        except Exception as e:
            self.logger.error("face_storage_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the storage writer and flush remaining observations."""
        if not self._writing:
            return True
        
        self._writing = False
        
        # Flush remaining queue
        try:
            while True:
                observation = self._write_queue.get_nowait()
                self._write_observation(observation)
        except Empty:
            pass
        
        # Wait for writer thread
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None
        
        # Close file
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
        
        self.logger.info("face_storage_stopped", 
                        observations_written=self._observations_written,
                        write_errors=self._write_errors)
        return True

    def write_observation(self, observation) -> bool:
        """
        Queue an observation for writing.
        
        Args:
            observation: ProcessedFaceObservation
            
        Returns:
            True if queued successfully
        """
        if not self._writing:
            return False
        
        try:
            self._write_queue.put_nowait(observation)
            return True
        except Exception as e:
            self._write_errors += 1
            self.logger.warning("face_storage_queue_failed", error=str(e))
            return False

    def _write_loop(self) -> None:
        """Background write loop."""
        while self._writing:
            try:
                observation = self._write_queue.get(timeout=0.1)
                self._write_observation(observation)
            except Empty:
                continue
            except Exception as e:
                self.logger.error("face_storage_write_loop_error", error=str(e))

    def _write_observation(self, observation) -> None:
        """Write a single face observation to JSONL file."""
        try:
            # Convert to dict with proper serialization
            data = self._observation_to_dict(observation)
            
            # Write JSONL line
            line = orjson.dumps(data) + b"\n"
            self._file_handle.write(line)
            self._file_handle.flush()
            
            self._observations_written += 1
            
        except Exception as e:
            self._write_errors += 1
            self.logger.error("face_storage_write_failed", 
                            observation_id=observation.observation_id, 
                            error=str(e))

    def _observation_to_dict(self, observation) -> Dict[str, Any]:
        """Convert ProcessedFaceObservation to dict with proper serialization."""
        # Helper to convert numpy types
        def convert_value(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {k: convert_value(v2) for k, v2 in v.items()}
            elif isinstance(v, list):
                return [convert_value(v2) for v2 in v]
            return v
        
        # Build the dict
        data = {
            "observation_id": observation.observation_id,
            "timestamp_monotonic": float(observation.timestamp_monotonic),
            "timestamp_wallclock": observation.timestamp_wallclock,
            "frame_index": int(observation.frame_index),
            "face_detected": bool(observation.face_detected),
            "detection_confidence": float(observation.detection_confidence) if observation.detection_confidence is not None else 0.0,
            "processing_time_ms": float(observation.processing_time_ms),
        }
        
        # Add optional fields
        if observation.bounding_box is not None:
            data["bounding_box"] = {k: int(v) for k, v in observation.bounding_box.items()}
        
        if observation.landmarks_3d is not None:
            data["landmarks_3d"] = [
                {"x": float(lm["x"]), "y": float(lm["y"]), "z": float(lm["z"])}
                for lm in observation.landmarks_3d
            ]
        
        if observation.landmarks_2d is not None:
            data["landmarks_2d"] = [
                {"x": int(lm["x"]), "y": int(lm["y"])}
                for lm in observation.landmarks_2d
            ]
        
        if observation.blendshapes is not None:
            data["blendshapes"] = {k: float(v) for k, v in observation.blendshapes.items()}
        
        if observation.iris is not None:
            data["iris"] = {
                "left": {k: float(v) for k, v in observation.iris["left"].items()},
                "right": {k: float(v) for k, v in observation.iris["right"].items()}
            }
        
        if observation.head_pose is not None:
            data["head_pose"] = {k: float(v) for k, v in observation.head_pose.items()}
        
        return convert_value(data)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "observations_written": self._observations_written,
            "write_errors": self._write_errors,
            "queue_size": self._write_queue.qsize(),
            "file_path": str(self._file_path),
        }