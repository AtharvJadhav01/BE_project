"""
Audio capture modality implementation.

Implements the AudioModalityInterface using sounddevice for microphone capture.
"""

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from core.events import Event, EventType, ModalityType, create_audio_event
from core.interfaces import AudioModalityInterface, ModalityStatus
from core.logging import get_logger
from core.timestamps import TimestampGenerator

from .listener import AudioListener
from .processor import AudioProcessor
from .storage import AudioStorage


class AudioCapture(AudioModalityInterface):
    """
    Audio capture modality.
    
    Captures microphone audio using sounddevice, processes chunks,
    and stores them as WAV files with metadata index.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Components
        self._listener: Optional[AudioListener] = None
        self._processor: Optional[AudioProcessor] = None
        self._storage: Optional[AudioStorage] = None
        
        # Session state
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._timestamp_generator: Optional[TimestampGenerator] = None
        self._unified_event_callback: Optional[callable] = None
        
        # Thread safety
        self._lock = threading.Lock()

    def set_unified_event_callback(self, callback: callable) -> None:
        """Set callback for unified event pipeline."""
        self._unified_event_callback = callback

    def _on_audio_chunk(self, chunk_index: int, data, start_time: float, duration: float, rms_db: float) -> None:
        """Callback for writing audio chunks to storage."""
        if self._storage:
            self._storage.write_chunk(chunk_index, data, start_time, duration, rms_db)

    def _on_audio_event(self, event: Event) -> None:
        """Callback for unified event pipeline."""
        if self._unified_event_callback:
            self._unified_event_callback(event)

    def start(self, session_id: str, session_dir: Optional[Path] = None) -> bool:
        """
        Start audio capture for a session.
        
        Args:
            session_id: Unique session identifier
            session_dir: Optional session directory path
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._status == ModalityStatus.RUNNING:
                self.logger.warning("audio_already_running", session_id=session_id)
                return True
            
            self._session_id = session_id
            
            # Use provided session_dir or construct from config
            if session_dir is not None:
                self._session_dir = session_dir
            else:
                self._session_dir = Path(self.config.get("storage", {}).get("output_directory", "data/sessions")) / session_id
            
            self._timestamp_generator = TimestampGenerator()
            
            try:
                # Check device availability before starting
                if not AudioListener.check_device_availability(self.config.get("device")):
                    self.logger.error("audio_no_input_device")
                    self._set_error(RuntimeError("No audio input device available"), "No audio input device available")
                    return False
                
                # Initialize components
                self._processor = AudioProcessor(self.config)
                self._processor.set_storage_callback(self._on_audio_chunk)
                self._processor.set_unified_event_callback(self._on_audio_event)
                
                self._storage = AudioStorage(self._session_dir, self.config, self._timestamp_generator)
                
                self._listener = AudioListener(
                    session_id=session_id,
                    timestamp_generator=self._timestamp_generator,
                    event_callback=self._on_audio_event,
                    config=self.config
                )
                self._listener.set_storage_callback(self._processor.process_chunk)
                
                # Start components in order
                if not self._storage.start():
                    raise RuntimeError("Failed to start audio storage")
                
                if not self._processor.start():
                    raise RuntimeError("Failed to start audio processor")
                
                if not self._listener.start():
                    raise RuntimeError("Failed to start audio listener")
                
                self._set_status(ModalityStatus.RUNNING)
                self.logger.info("audio_capture_started", 
                               session_id=session_id,
                               sample_rate=self.config.get("sample_rate", 16000),
                               channels=self.config.get("channels", 1))
                return True
                
            except Exception as e:
                self._set_error(e, f"Failed to start audio capture: {e}")
                self._cleanup()
                return False

    def stop(self) -> bool:
        """
        Stop audio capture.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if self._status == ModalityStatus.STOPPED:
                return True
            
            self._set_status(ModalityStatus.STOPPING)
            
            success = True
            
            # Stop in reverse order
            if self._listener:
                if not self._listener.stop():
                    success = False
            
            if self._processor:
                if not self._processor.stop():
                    success = False
            
            if self._storage:
                if not self._storage.stop():
                    success = False
            
            self._cleanup()
            
            if success:
                self._set_status(ModalityStatus.STOPPED)
                self.logger.info("audio_capture_stopped", session_id=self._session_id)
            else:
                self._set_status(ModalityStatus.ERROR)
                self.logger.error("audio_capture_stop_failed", session_id=self._session_id)
            
            return success

    def pause(self) -> bool:
        """
        Pause audio capture.
        
        Returns:
            True if paused successfully, False otherwise
        """
        with self._lock:
            if self._status != ModalityStatus.RUNNING:
                return False
            
            if self._listener:
                result = self._listener.pause()
                if result:
                    self._set_status(ModalityStatus.PAUSED)
                    self.logger.info("audio_capture_paused", session_id=self._session_id)
                return result
            return False

    def resume(self) -> bool:
        """
        Resume audio capture.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        with self._lock:
            if self._status != ModalityStatus.PAUSED:
                return False
            
            if self._listener:
                result = self._listener.resume()
                if result:
                    self._set_status(ModalityStatus.RUNNING)
                    self.logger.info("audio_capture_resumed", session_id=self._session_id)
                return result
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._listener = None
        self._processor = None
        self._storage = None
        self._session_id = None
        self._session_dir = None
        self._timestamp_generator = None

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        stats = {
            "status": self._status.value,
            "session_id": self._session_id,
        }
        
        if self._listener:
            stats["listener"] = {
                "running": self._listener.is_running,
                "paused": self._listener.is_paused,
            }
        
        if self._processor:
            stats["processor"] = self._processor.get_stats()
        
        if self._storage:
            stats["storage"] = self._storage.get_stats()
        
        return stats