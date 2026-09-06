"""
Audio processor for handling audio chunks.

Processes audio chunks from listener and forwards to storage/unified pipeline.
"""

import threading
import queue
import numpy as np
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass
from queue import Empty

from core.logging import get_logger


@dataclass
class ProcessedAudioChunk:
    """Processed audio chunk ready for storage."""
    chunk_index: int
    data: np.ndarray
    start_time_monotonic: float
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_db: float


class AudioProcessor:
    """
    Processes audio chunks from the listener.
    
    Handles validation, basic analysis, and forwarding to storage.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Audio settings
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        
        # Event queue for async processing
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        self._processing = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._storage_callback: Optional[Callable] = None
        self._unified_event_callback: Optional[Callable] = None
        
        # Statistics
        self._chunks_processed = 0
        self._chunks_dropped = 0

    def set_storage_callback(self, callback: Callable) -> None:
        """Set callback for storing processed chunks."""
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
        self.logger.info("audio_processor_started")
        return True

    def stop(self) -> bool:
        """Stop the processor."""
        if not self._processing:
            return True
        
        self._processing = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self.logger.info("audio_processor_stopped", 
                        processed=self._chunks_processed, 
                        dropped=self._chunks_dropped)
        return True

    def process_chunk(self, chunk_index: int, data: np.ndarray, 
                     start_time: float, duration: float, rms_db: float) -> bool:
        """
        Queue an audio chunk for processing.
        
        Args:
            chunk_index: Sequential chunk index
            data: Audio data as numpy array
            start_time: Monotonic start time
            duration: Duration in seconds
            rms_db: RMS level in dBFS
            
        Returns:
            True if queued successfully, False if queue full
        """
        if not self._processing:
            return False
        
        try:
            chunk = ProcessedAudioChunk(
                chunk_index=chunk_index,
                data=data.copy(),  # Copy to avoid buffer reuse issues
                start_time_monotonic=start_time,
                duration_seconds=duration,
                sample_rate=self.sample_rate,
                channels=self.channels,
                rms_db=rms_db,
            )
            self._chunk_queue.put_nowait(chunk)
            return True
        except queue.Full:
            self._chunks_dropped += 1
            self.logger.warning("audio_processor_queue_full", dropped=self._chunks_dropped)
            return False
        except Exception as e:
            self.logger.error("audio_processor_enqueue_failed", error=str(e))
            return False

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._processing:
            try:
                chunk = self._chunk_queue.get(timeout=0.1)
                self._process_single_chunk(chunk)
            except Empty:
                continue
            except Exception as e:
                self.logger.error("audio_processor_error", error=str(e))

    def _process_single_chunk(self, chunk: ProcessedAudioChunk) -> None:
        """Process a single audio chunk."""
        try:
            # Forward to storage with individual arguments
            if self._storage_callback:
                self._storage_callback(
                    chunk.chunk_index,
                    chunk.data,
                    chunk.start_time_monotonic,
                    chunk.duration_seconds,
                    chunk.rms_db
                )
            
            # Forward to unified event pipeline (for sync events)
            if self._unified_event_callback:
                from core.events import create_audio_event, EventType
                from core.timestamps import TimestampGenerator
                
                # Create a timestamp generator for this chunk
                gen = TimestampGenerator()
                # We need to use the session's timestamp generator ideally
                # For now, we'll just log that the chunk was processed
                pass
            
            self._chunks_processed += 1
            
        except Exception as e:
            self.logger.error("audio_chunk_processing_failed", 
                            chunk_index=chunk.chunk_index, 
                            error=str(e))

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "chunks_processed": self._chunks_processed,
            "chunks_dropped": self._chunks_dropped,
            "queue_size": self._chunk_queue.qsize(),
        }


@dataclass
class ProcessedAudioChunk:
    """Processed audio chunk ready for storage."""
    chunk_index: int
    data: np.ndarray
    start_time_monotonic: float
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_db: float