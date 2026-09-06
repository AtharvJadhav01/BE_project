"""
Audio storage for writing audio chunks to WAV files.

Writes audio chunks as WAV files and maintains an index file (audio_index.jsonl).
"""

import threading
import orjson
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from queue import Queue, Empty

from core.logging import get_logger
from core.timestamps import TimestampGenerator


class AudioStorage:
    """
    Stores audio chunks to WAV files and maintains index.
    
    Uses async writing with a background thread for performance.
    """

    def __init__(self, session_dir: Path, config: Dict[str, Any], 
                 timestamp_generator: Optional[TimestampGenerator] = None):
        self.session_dir = session_dir
        self.config = config
        self.logger = get_logger(__name__)
        
        # Audio settings
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.format = config.get("format", "wav")
        
        # Timestamp generator for wallclock conversion
        self._timestamp_generator = timestamp_generator
        
        # File paths
        self._audio_dir = session_dir / "audio"
        self._index_path = self._audio_dir / "audio_index.jsonl"
        
        # Async writing
        self._write_queue: Queue = Queue()
        self._writing = False
        self._writer_thread: Optional[threading.Thread] = None
        self._index_handle = None
        
        # Statistics
        self._chunks_written = 0
        self._write_errors = 0

    def start(self) -> bool:
        """Start the storage writer."""
        if self._writing:
            return True
        
        try:
            # Ensure directory exists
            self._audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Open index file for appending
            self._index_handle = open(self._index_path, "ab")
            
            self._writing = True
            self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self._writer_thread.start()
            
            self.logger.info("audio_storage_started", 
                           audio_dir=str(self._audio_dir),
                           index_path=str(self._index_path))
            return True
        except Exception as e:
            self.logger.error("audio_storage_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the storage writer and flush remaining chunks."""
        if not self._writing:
            return True
        
        self._writing = False
        
        # Flush remaining queue
        try:
            while True:
                chunk = self._write_queue.get_nowait()
                self._write_chunk(chunk)
        except Empty:
            pass
        
        # Wait for writer thread
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None
        
        # Close index file
        if self._index_handle:
            try:
                self._index_handle.close()
            except Exception:
                pass
            self._index_handle = None
        
        self.logger.info("audio_storage_stopped", 
                        chunks_written=self._chunks_written,
                        write_errors=self._write_errors)
        return True

    def write_chunk(self, chunk_index: int, data: np.ndarray, 
                   start_time: float, duration: float, rms_db: float) -> bool:
        """
        Queue a chunk for writing.
        
        Args:
            chunk_index: Sequential chunk index
            data: Audio data as numpy array
            start_time: Monotonic start time in seconds
            duration: Duration in seconds
            rms_db: RMS level in dBFS
            
        Returns:
            True if queued successfully
        """
        if not self._writing:
            return False
        
        try:
            chunk_data = {
                "chunk_index": chunk_index,
                "data": data.copy(),  # Copy to avoid buffer reuse
                "start_time": start_time,
                "duration": duration,
                "rms_db": rms_db,
            }
            self._write_queue.put_nowait(chunk_data)
            return True
        except Exception as e:
            self._write_errors += 1
            self.logger.warning("audio_storage_queue_failed", error=str(e))
            return False

    def _write_loop(self) -> None:
        """Background write loop."""
        while self._writing:
            try:
                chunk = self._write_queue.get(timeout=0.1)
                self._write_chunk(chunk)
            except Empty:
                continue
            except Exception as e:
                self.logger.error("audio_storage_write_loop_error", error=str(e))

    def _write_chunk(self, chunk: Dict[str, Any]) -> None:
        """Write a single audio chunk to WAV file and update index."""
        chunk_index = chunk["chunk_index"]
        data = chunk["data"]
        start_time = chunk["start_time"]
        duration = chunk["duration"]
        rms_db = chunk["rms_db"]
        
        try:
            # Generate filename
            filename = f"chunk_{chunk_index:06d}.wav"
            filepath = self._audio_dir / filename
            
            # Write WAV file
            # soundfile expects (frames, channels) for mono, or (frames,) for mono
            if self.channels == 1:
                write_data = data[:, 0] if data.ndim > 1 else data
            else:
                write_data = data
            
            sf.write(
                filepath,
                write_data,
                self.sample_rate,
                subtype='PCM_16'
            )
            
            # Calculate metadata
            sample_count = len(write_data)
            bytes_written = filepath.stat().st_size
            
            # Create index entry
            end_time = start_time + duration
            
            # Convert monotonic to wallclock if timestamp generator available
            start_wallclock = ""
            end_wallclock = ""
            if self._timestamp_generator:
                try:
                    start_wallclock = self._timestamp_generator.monotonic_to_wallclock(start_time)
                    end_wallclock = self._timestamp_generator.monotonic_to_wallclock(end_time)
                except Exception:
                    pass  # Leave empty if conversion fails
            
            index_entry = {
                "chunk_index": chunk_index,
                "filename": filename,
                "start_time_monotonic": float(start_time),
                "end_time_monotonic": float(end_time),
                "start_time_wallclock": start_wallclock,
                "end_time_wallclock": end_wallclock,
                "duration_ms": int(duration * 1000),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_count": int(sample_count),
                "bytes": int(bytes_written),
                "rms_db": float(rms_db),
            }
            
            # Write index entry
            line = orjson.dumps(index_entry) + b"\n"
            self._index_handle.write(line)
            self._index_handle.flush()
            
            self._chunks_written += 1
            
        except Exception as e:
            self._write_errors += 1
            self.logger.error("audio_storage_write_failed", 
                            chunk_index=chunk_index, 
                            error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "chunks_written": self._chunks_written,
            "write_errors": self._write_errors,
            "queue_size": self._write_queue.qsize(),
            "audio_dir": str(self._audio_dir),
            "index_path": str(self._index_path),
        }