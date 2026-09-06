"""
Audio listener using sounddevice for microphone capture.

Captures audio from microphone using callback-based streaming.
"""

import threading
import queue
import numpy as np
import sounddevice as sd
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass

from core.events import Event, EventType, ModalityType, create_audio_event
from core.logging import get_logger
from core.timestamps import TimestampGenerator


@dataclass
class AudioChunk:
    """Raw audio chunk data from sounddevice."""
    data: np.ndarray
    timestamp: float  # monotonic time
    frames: int


class AudioListener:
    """
    Microphone listener using sounddevice.
    
    Uses callback-based streaming for real-time capture with low latency.
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
        
        # Audio settings
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.chunk_duration_ms = config.get("chunk_duration_ms", 10000)
        self.device = config.get("device", None)
        
        # Calculate frames per chunk
        self.frames_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # State
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        
        # Audio buffer queue
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Worker thread for processing audio chunks
        self._worker_thread: Optional[threading.Thread] = None
        
        # Chunk tracking
        self._chunk_index = 0
        self._chunk_start_time: Optional[float] = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        """
        Sounddevice callback for incoming audio data.
        
        Called from audio thread - must be fast and non-blocking.
        """
        if not self._running or self._paused:
            return
        
        if status:
            self.logger.warning("audio_callback_status", status=str(status))
        
        # Copy data to avoid issues with buffer reuse
        chunk_data = indata.copy()
        current_time = self.timestamp_generator.now_monotonic()
        
        try:
            self._audio_queue.put_nowait(AudioChunk(
                data=chunk_data,
                timestamp=current_time,
                frames=frames
            ))
        except queue.Full:
            self.logger.warning("audio_queue_full_dropping_chunk")

    def _worker_loop(self) -> None:
        """Worker thread to process audio chunks from queue."""
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                self._process_chunk(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("audio_worker_error", error=str(e))

    def _process_chunk(self, chunk: AudioChunk) -> None:
        """Process a single audio chunk and emit events."""
        with self._lock:
            if not self._running:
                return
            
            self._chunk_index += 1
            chunk_start = chunk.timestamp
            chunk_duration = len(chunk.data) / self.sample_rate
            chunk_end = chunk_start + chunk_duration
            rms_db = self._calculate_rms_db(chunk.data)
            
            # Emit chunk start event
            start_event = create_audio_event(
                session_id=self.session_id,
                event_type=EventType.AUDIO_CHUNK_START,
                data={
                    "chunk_index": self._chunk_index,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                },
                timestamp_generator=self.timestamp_generator,
            )
            self.event_callback(start_event)
            
            # Store chunk for storage writer
            if hasattr(self, '_storage_callback') and self._storage_callback:
                self._storage_callback(self._chunk_index, chunk.data, chunk_start, chunk_duration, rms_db)
            
            # Emit chunk end event
            end_event = create_audio_event(
                session_id=self.session_id,
                event_type=EventType.AUDIO_CHUNK_END,
                data={
                    "chunk_index": self._chunk_index,
                    "duration_ms": int(chunk_duration * 1000),
                    "rms_db": rms_db,
                },
                timestamp_generator=self.timestamp_generator,
            )
            self.event_callback(end_event)

    def _calculate_rms_db(self, audio_data: np.ndarray) -> float:
        """Calculate RMS level in dBFS."""
        if len(audio_data) == 0:
            return -120.0  # silence
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        if rms == 0:
            return -120.0
        return 20 * np.log10(rms)

    def set_storage_callback(self, callback: Callable) -> None:
        """Set callback for storing audio chunks."""
        self._storage_callback = callback

    def start(self) -> bool:
        """Start the audio listener."""
        if self._running:
            return True
        
        try:
            # Check device availability
            devices = sd.query_devices()
            if self.device is not None and self.device >= len(devices):
                self.logger.error("audio_device_not_found", device=self.device)
                return False
            
            # Check if any input device is available
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                self.logger.error("no_audio_input_devices")
                return False
            
            # Create input stream
            self._stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=1024,  # Small blocksize for low latency
                dtype='float32'
            )
            
            self._stream.start()
            self._running = True
            self._paused = False
            self._chunk_index = 0
            
            # Start worker thread
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            
            self.logger.info("audio_listener_started", 
                           session_id=self.session_id,
                           sample_rate=self.sample_rate,
                           channels=self.channels)
            return True
            
        except Exception as e:
            self.logger.error("audio_listener_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the audio listener."""
        if not self._running:
            return True
        
        try:
            self._running = False
            
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            
            if self._worker_thread:
                self._worker_thread.join(timeout=2.0)
                self._worker_thread = None
            
            # Clear queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("audio_listener_stopped", session_id=self.session_id)
            return True
            
        except Exception as e:
            self.logger.error("audio_listener_stop_failed", error=str(e))
            return False

    def pause(self) -> bool:
        """Pause the audio listener."""
        if not self._running or self._paused:
            return False
        
        with self._lock:
            self._paused = True
            self.logger.info("audio_listener_paused", session_id=self.session_id)
            return True

    def resume(self) -> bool:
        """Resume the audio listener."""
        if not self._running or not self._paused:
            return False
        
        with self._lock:
            self._paused = False
            self.logger.info("audio_listener_resumed", session_id=self.session_id)
            return True

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    @staticmethod
    def get_available_devices() -> List[Dict[str, Any]]:
        """Get list of available audio input devices."""
        devices = sd.query_devices()
        return [
            {
                "index": i,
                "name": d["name"],
                "max_input_channels": d["max_input_channels"],
                "default_samplerate": d["default_samplerate"],
            }
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    @staticmethod
    def check_device_availability(device: Optional[int] = None) -> bool:
        """Check if a specific device or any input device is available."""
        try:
            devices = sd.query_devices()
            if device is not None:
                return 0 <= device < len(devices) and devices[device]["max_input_channels"] > 0
            return any(d["max_input_channels"] > 0 for d in devices)
        except Exception:
            return False