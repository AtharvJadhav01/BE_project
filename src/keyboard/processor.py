"""
Keyboard event processor.

Processes raw keyboard events and prepares them for storage.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from queue import Queue, Empty

from core.events import Event, EventType, ModalityType
from core.logging import get_logger


@dataclass
class ProcessedKeyboardEvent:
    """Processed keyboard event ready for storage."""
    event_id: str
    timestamp_monotonic: float
    timestamp_wallclock: str
    event_type: str  # "key_down" or "key_up"
    key: str
    key_code: int
    modifiers: List[str]
    duration_ms: Optional[float]
    is_printable: bool
    is_sensitive_context: bool
    window_title: Optional[str]
    process_name: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL storage."""
        return {
            "event_id": self.event_id,
            "timestamp_monotonic": self.timestamp_monotonic,
            "timestamp_wallclock": self.timestamp_wallclock,
            "event_type": self.event_type,
            "key": self.key,
            "key_code": self.key_code,
            "modifiers": self.modifiers,
            "duration_ms": self.duration_ms,
            "is_printable": self.is_printable,
            "is_sensitive_context": self.is_sensitive_context,
            "window_title": self.window_title,
            "process_name": self.process_name,
        }


class KeyboardProcessor:
    """
    Processes keyboard events from the listener.
    
    Handles event validation, enrichment, and forwarding to storage.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Event queue for async processing
        self._event_queue: Queue = Queue()
        self._processing = False
        self._worker_thread: Optional[threading.Thread] = None
        self._storage_callback: Optional[callable] = None
        self._unified_event_callback: Optional[callable] = None
        self._analysis_callback: Optional[callable] = None
        
        # Statistics
        self._events_processed = 0
        self._events_dropped = 0

    def set_storage_callback(self, callback: callable) -> None:
        """Set callback for storing processed events."""
        self._storage_callback = callback

    def set_unified_event_callback(self, callback: callable) -> None:
        """Set callback for unified event pipeline."""
        self._unified_event_callback = callback

    def set_analysis_callback(self, callback: callable) -> None:
        """Set callback for keyboard analysis."""
        self._analysis_callback = callback

    def start(self) -> bool:
        """Start the processor."""
        if self._processing:
            return True
        
        self._processing = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        self.logger.info("keyboard_processor_started")
        return True

    def stop(self) -> bool:
        """Stop the processor."""
        if not self._processing:
            return True
        
        self._processing = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self.logger.info("keyboard_processor_stopped", 
                        processed=self._events_processed, 
                        dropped=self._events_dropped)
        return True

    def process_event(self, event: Event) -> bool:
        """
        Queue an event for processing.
        
        Args:
            event: Raw keyboard event from listener
            
        Returns:
            True if queued successfully, False if queue full
        """
        if not self._processing:
            return False
        
        try:
            self._event_queue.put_nowait(event)
            return True
        except Exception:
            self._events_dropped += 1
            self.logger.warning("keyboard_event_queue_full", dropped=self._events_dropped)
            return False

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._processing:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._process_single_event(event)
            except Empty:
                continue
            except Exception as e:
                self.logger.error("keyboard_processor_error", error=str(e))

    def _process_single_event(self, event: Event) -> None:
        """Process a single keyboard event."""
        try:
            data = event.data
            
            # Validate required fields
            if "key" not in data or "event_type" not in data:
                self.logger.warning("keyboard_event_missing_fields", event_id=event.event_id)
                return
            
            # Create processed event
            processed = ProcessedKeyboardEvent(
                event_id=event.event_id,
                timestamp_monotonic=event.timestamp.monotonic,
                timestamp_wallclock=event.timestamp.wallclock,
                event_type=data.get("event_type", "unknown"),
                key=data.get("key", ""),
                key_code=data.get("key_code", 0),
                modifiers=data.get("modifiers", []),
                duration_ms=data.get("duration_ms"),
                is_printable=data.get("is_printable", False),
                is_sensitive_context=data.get("is_sensitive_context", False),
                window_title=data.get("window_title"),
                process_name=data.get("process_name"),
            )
            
            # Forward to storage
            if self._storage_callback:
                self._storage_callback(processed)
            
            # Forward to analysis callback
            if self._analysis_callback:
                self._analysis_callback(event)
            
            # Forward to unified event pipeline
            if self._unified_event_callback:
                self._unified_event_callback(event)
            
            self._events_processed += 1
            
        except Exception as e:
            self.logger.error("keyboard_event_processing_failed", 
                            event_id=event.event_id, 
                            error=str(e))

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "events_processed": self._events_processed,
            "events_dropped": self._events_dropped,
            "queue_size": self._event_queue.qsize(),
        }