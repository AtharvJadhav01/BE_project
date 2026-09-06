"""
Keyboard event storage.

Writes keyboard events to keystrokes.jsonl in the session directory.
"""

import threading
import orjson
from pathlib import Path
from typing import Any, Dict, Optional
from queue import Queue, Empty

from core.logging import get_logger

from .processor import ProcessedKeyboardEvent


class KeyboardStorage:
    """
    Stores keyboard events to JSONL file.
    
    Uses async writing with a background thread for performance.
    """

    def __init__(self, session_dir: Path, config: Dict[str, Any]):
        self.session_dir = session_dir
        self.config = config
        self.logger = get_logger(__name__)
        
        # File path
        self._file_path = session_dir / "keystrokes.jsonl"
        
        # Async writing
        self._write_queue: Queue = Queue()
        self._writing = False
        self._writer_thread: Optional[threading.Thread] = None
        self._file_handle = None
        
        # Statistics
        self._events_written = 0
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
            
            self.logger.info("keyboard_storage_started", path=str(self._file_path))
            return True
        except Exception as e:
            self.logger.error("keyboard_storage_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the storage writer and flush remaining events."""
        if not self._writing:
            return True
        
        self._writing = False
        
        # Flush remaining queue
        try:
            while True:
                event = self._write_queue.get_nowait()
                self._write_event(event)
        except Empty:
            pass
        
        # Wait for writer thread
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None
        
        # Close file
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
        
        self.logger.info("keyboard_storage_stopped", 
                        events_written=self._events_written,
                        write_errors=self._write_errors)
        return True

    def write_event(self, event: ProcessedKeyboardEvent) -> bool:
        """
        Queue an event for writing.
        
        Args:
            event: Processed keyboard event
            
        Returns:
            True if queued successfully
        """
        if not self._writing:
            return False
        
        try:
            self._write_queue.put_nowait(event)
            return True
        except Exception:
            self._write_errors += 1
            self.logger.warning("keyboard_storage_queue_full")
            return False

    def _write_loop(self) -> None:
        """Background write loop."""
        while self._writing:
            try:
                event = self._write_queue.get(timeout=0.1)
                self._write_event(event)
            except Empty:
                continue
            except Exception as e:
                self.logger.error("keyboard_storage_write_loop_error", error=str(e))

    def _write_event(self, event: ProcessedKeyboardEvent) -> None:
        """Write a single event to file."""
        if not self._file_handle:
            return
        
        try:
            # Convert to JSON line
            line = orjson.dumps(event.to_dict()) + b"\n"
            self._file_handle.write(line)
            self._file_handle.flush()
            self._events_written += 1
        except Exception as e:
            self._write_errors += 1
            self.logger.error("keyboard_storage_write_failed", 
                            event_id=event.event_id, 
                            error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "events_written": self._events_written,
            "write_errors": self._write_errors,
            "queue_size": self._write_queue.qsize(),
            "file_path": str(self._file_path),
        }