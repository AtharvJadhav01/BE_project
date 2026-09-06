"""
Synchronization module for unified event handling across modalities.

Provides event buffering, ordering, and unified event stream writing.
"""

import threading
import queue
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue

from core.events import Event, EventType, ModalityType, create_system_event
from core.logging import get_logger
from core.timestamps import TimestampGenerator


@dataclass
class EventBufferEntry:
    """Entry in the event buffer with ordering key."""
    sort_key: float  # monotonic timestamp for sorting
    event: Event
    arrival_time: float = field(default_factory=time.time)

    def __lt__(self, other: "EventBufferEntry") -> bool:
        """Compare for priority queue ordering."""
        return self.sort_key < other.sort_key


class EventSynchronizer:
    """
    Handles synchronization of events from multiple modalities.
    
    Features:
    - Buffers events from multiple modalities
    - Orders events by monotonic timestamp
    - Handles out-of-order/delayed events
    - Flushes events to unified event stream (events.jsonl)
    - Maintains event ordering guarantees
    """
    
    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        timestamp_generator: TimestampGenerator,
        config: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.session_dir = session_dir
        self.timestamp_generator = timestamp_generator
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Buffer for out-of-order events
        self._buffer: PriorityQueue = PriorityQueue()
        self._buffer_lock = threading.Lock()
        
        # Expected next timestamp for each modality (for gap detection)
        self._expected_timestamps: Dict[str, float] = {}
        
        # Flush configuration
        self._flush_interval_ms = self.config.get("flush_interval_ms", 100)
        self._max_buffer_age_ms = self.config.get("max_buffer_age_ms", 500)
        self._max_buffer_size = self.config.get("max_buffer_size", 1000)
        
        # Output file
        self._events_file = session_dir / "events.jsonl"
        self._file_handle = None
        
        # Flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Statistics
        self._events_buffered = 0
        self._events_flushed = 0
        self._events_dropped = 0
        self._out_of_order_count = 0

    def start(self) -> bool:
        """Start the synchronizer."""
        try:
            # Ensure directory exists
            self.session_dir.mkdir(parents=True, exist_ok=True)
            
            # Open events file with unbuffered mode (0 = unbuffered)
            # This helps with Windows file locking issues
            self._file_handle = open(self._events_file, "ab", buffering=0)
            
            # Start flush thread
            self._running = True
            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._flush_thread.start()
            
            self.logger.info("event_synchronizer_started", 
                           session_id=self.session_id,
                           events_file=str(self._events_file))
            return True
        except Exception as e:
            self.logger.error("event_synchronizer_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the synchronizer and flush remaining events."""
        self._running = False
        
        # Wait for flush thread to finish first
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        
        # Now flush any remaining buffer (no thread should be writing)
        self._flush_buffer()
        
        # Close file - ensure it's closed on Windows
        if self._file_handle:
            try:
                self._file_handle.flush()
                self._file_handle.close()
            except Exception:
                pass
            finally:
                self._file_handle = None
        
        # Small delay to ensure Windows releases file lock
        time.sleep(0.1)
        
        self.logger.info("event_synchronizer_stopped",
                        events_buffered=self._events_buffered,
                        events_flushed=self._events_flushed,
                        events_dropped=self._events_dropped,
                        out_of_order=self._out_of_order_count)
        return True

    def add_event(self, event: Event) -> bool:
        """
        Add an event to the synchronization buffer.
        
        Events are buffered and ordered by monotonic timestamp before flushing.
        
        Args:
            event: Event to add to the synchronized stream
            
        Returns:
            True if added successfully, False if buffer full
        """
        if not self._running:
            return False
        
        try:
            # Use monotonic timestamp as sort key
            sort_key = event.timestamp.monotonic
            
            entry = EventBufferEntry(
                sort_key=sort_key,
                event=event
            )
            
            with self._buffer_lock:
                if self._buffer.qsize() >= self._max_buffer_size:
                    self._events_dropped += 1
                    self.logger.warning("event_buffer_full", 
                                      session_id=self.session_id,
                                      dropped=self._events_dropped)
                    return False
                
                self._buffer.put(entry)
                self._events_buffered += 1
                
                # Track expected timestamps per modality
                modality = event.modality.value
                if modality in self._expected_timestamps:
                    expected = self._expected_timestamps[modality]
                    if sort_key < expected:
                        self._out_of_order_count += 1
                        self.logger.debug("out_of_order_event",
                                        modality=modality,
                                        expected=expected,
                                        actual=sort_key)
                self._expected_timestamps[modality] = sort_key
            
            return True
        except Exception as e:
            self.logger.error("add_event_failed", error=str(e))
            return False

    def _flush_loop(self) -> None:
        """Background loop to periodically flush buffered events."""
        while self._running:
            try:
                self._flush_buffer()
                time.sleep(self._flush_interval_ms / 1000.0)
            except Exception as e:
                self.logger.error("flush_loop_error", error=str(e))
                time.sleep(0.1)

    def _flush_buffer(self) -> None:
        """Flush events from buffer to file in timestamp order."""
        if not self._file_handle:
            return
        
        # Get current time for age-based flushing
        current_time = time.time()
        flush_threshold = current_time - (self._max_buffer_age_ms / 1000.0)
        
        events_to_flush = []
        
        with self._buffer_lock:
            # Peek at buffer to find events ready to flush
            # We flush events that are older than max_buffer_age_ms
            # or if buffer is getting full
            buffer_size = self._buffer.qsize()
            should_flush_all = buffer_size >= self._max_buffer_size * 0.8
            
            # Collect events to flush
            temp_buffer = []
            while not self._buffer.empty():
                try:
                    entry = self._buffer.get_nowait()
                    if should_flush_all or entry.arrival_time <= flush_threshold:
                        events_to_flush.append(entry)
                    else:
                        temp_buffer.append(entry)
                except Empty:
                    break
            
            # Put back events not ready to flush
            for entry in temp_buffer:
                self._buffer.put(entry)
        
        # Sort events by timestamp before writing
        events_to_flush.sort(key=lambda e: e.sort_key)
        
        # Write events to file
        for entry in events_to_flush:
            try:
                self._write_event(entry.event)
                self._events_flushed += 1
            except Exception as e:
                self.logger.error("flush_event_failed", 
                                event_id=entry.event.event_id,
                                error=str(e))

    def _write_event(self, event: Event) -> None:
        """Write a single event to the events.jsonl file."""
        if not self._file_handle:
            return
        
        import orjson
        line = orjson.dumps(event.to_dict()) + b"\n"
        self._file_handle.write(line)
        self._file_handle.flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        with self._buffer_lock:
            buffer_size = self._buffer.qsize()
        
        return {
            "events_buffered": self._events_buffered,
            "events_flushed": self._events_flushed,
            "events_dropped": self._events_dropped,
            "out_of_order_count": self._out_of_order_count,
            "buffer_size": buffer_size,
            "buffer_utilization": buffer_size / self._max_buffer_size if self._max_buffer_size > 0 else 0,
        }

    def get_buffered_events(self, modality: Optional[str] = None) -> List[Event]:
        """Get currently buffered events (for testing/inspection)."""
        with self._buffer_lock:
            events = []
            temp = []
            while not self._buffer.empty():
                try:
                    entry = self._buffer.get_nowait()
                    temp.append(entry)
                    if modality is None or entry.event.modality.value == modality:
                        events.append(entry.event)
                except Empty:
                    break
            # Put back
            for entry in temp:
                self._buffer.put(entry)
        return events


class SynchronizationManager:
    """
    High-level synchronization manager that coordinates multiple sessions.
    
    Manages one EventSynchronizer per active session.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self._synchronizers: Dict[str, EventSynchronizer] = {}
        self._lock = threading.Lock()

    def create_synchronizer(
        self,
        session_id: str,
        session_dir: Path,
        timestamp_generator: TimestampGenerator
    ) -> EventSynchronizer:
        """Create and start a new event synchronizer for a session."""
        with self._lock:
            if session_id in self._synchronizers:
                self.logger.warning("synchronizer_exists", session_id=session_id)
                return self._synchronizers[session_id]
            
            synchronizer = EventSynchronizer(
                session_id=session_id,
                session_dir=session_dir,
                timestamp_generator=timestamp_generator,
                config=self.config.get("synchronization", {})
            )
            
            if synchronizer.start():
                self._synchronizers[session_id] = synchronizer
                return synchronizer
            else:
                raise RuntimeError(f"Failed to start synchronizer for session {session_id}")

    def remove_synchronizer(self, session_id: str) -> bool:
        """Remove and stop a synchronizer."""
        with self._lock:
            if session_id not in self._synchronizers:
                return False
            
            synchronizer = self._synchronizers.pop(session_id)
            synchronizer.stop()
            return True

    def get_synchronizer(self, session_id: str) -> Optional[EventSynchronizer]:
        """Get synchronizer for a session."""
        with self._lock:
            return self._synchronizers.get(session_id)

    def add_event(self, session_id: str, event: Event) -> bool:
        """Add an event to a session's synchronizer."""
        with self._lock:
            synchronizer = self._synchronizers.get(session_id)
            if not synchronizer:
                self.logger.warning("synchronizer_not_found", session_id=session_id)
                return False
            return synchronizer.add_event(event)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active synchronizers."""
        with self._lock:
            return {
                session_id: sync.get_stats()
                for session_id, sync in self._synchronizers.items()
            }

    def cleanup(self) -> None:
        """Stop all synchronizers and clean up."""
        with self._lock:
            for synchronizer in self._synchronizers.values():
                synchronizer.stop()
            self._synchronizers.clear()
            self.logger.info("synchronization_manager_cleaned_up")


# Global synchronization manager instance
_synchronization_manager: Optional[SynchronizationManager] = None


def get_synchronization_manager(config: Optional[Dict[str, Any]] = None) -> SynchronizationManager:
    """Get or create the global synchronization manager instance."""
    global _synchronization_manager
    if _synchronization_manager is None:
        _synchronization_manager = SynchronizationManager(config)
    return _synchronization_manager


def reset_synchronization_manager() -> None:
    """Reset the global synchronization manager instance."""
    global _synchronization_manager
    if _synchronization_manager is not None:
        _synchronization_manager.cleanup()
        _synchronization_manager = None