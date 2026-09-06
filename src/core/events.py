"""
Unified event structure for all modalities.

Defines the common event schema used across all modalities for storage and processing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

from .timestamps import Timestamp, TimestampGenerator


class ModalityType(str, Enum):
    """Type of modality that generated the event."""
    VOICE = "voice"
    KEYBOARD = "keyboard"
    FACE = "face"
    SYSTEM = "system"


class EventType(str, Enum):
    """Types of events that can be generated."""
    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_PAUSE = "session_pause"
    SESSION_RESUME = "session_resume"
    MODALITY_ERROR = "modality_error"
    MODALITY_STATUS_CHANGE = "modality_status_change"

    # Audio events
    AUDIO_CHUNK_START = "audio_chunk_start"
    AUDIO_CHUNK_END = "audio_chunk_end"
    VOICE_ACTIVITY = "voice_activity"

    # Keyboard events
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"

    # Face events
    FACE_OBSERVATION = "face_observation"
    FACE_LOST = "face_lost"
    FACE_FOUND = "face_found"


@dataclass
class Event:
    """
    Unified event structure for all modalities.

    All events from all modalities use this structure for consistent storage
    and cross-modal analysis.
    """
    session_id: str
    timestamp: Timestamp
    modality: ModalityType
    event_type: EventType
    data: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.to_dict(),
            "modality": self.modality.value,
            "event_type": self.event_type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict, timestamp_generator: Optional[TimestampGenerator] = None) -> "Event":
        """Create Event from dictionary."""
        timestamp = Timestamp.from_dict(data["timestamp"])
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            timestamp=timestamp,
            modality=ModalityType(data["modality"]),
            event_type=EventType(data["event_type"]),
            data=data["data"],
        )


def create_system_event(
    session_id: str,
    event_type: EventType,
    data: dict[str, Any],
    timestamp_generator: Optional[TimestampGenerator] = None
) -> Event:
    """Create a system event."""
    if timestamp_generator is None:
        timestamp_generator = TimestampGenerator()
    return Event(
        session_id=session_id,
        timestamp=timestamp_generator.now(),
        modality=ModalityType.SYSTEM,
        event_type=event_type,
        data=data,
    )


def create_audio_event(
    session_id: str,
    event_type: EventType,
    data: dict[str, Any],
    timestamp_generator: Optional[TimestampGenerator] = None
) -> Event:
    """Create an audio event."""
    if timestamp_generator is None:
        timestamp_generator = TimestampGenerator()
    return Event(
        session_id=session_id,
        timestamp=timestamp_generator.now(),
        modality=ModalityType.VOICE,
        event_type=event_type,
        data=data,
    )


def create_keyboard_event(
    session_id: str,
    event_type: EventType,
    data: dict[str, Any],
    timestamp_generator: Optional[TimestampGenerator] = None
) -> Event:
    """Create a keyboard event."""
    if timestamp_generator is None:
        timestamp_generator = TimestampGenerator()
    return Event(
        session_id=session_id,
        timestamp=timestamp_generator.now(),
        modality=ModalityType.KEYBOARD,
        event_type=event_type,
        data=data,
    )


def create_face_event(
    session_id: str,
    event_type: EventType,
    data: dict[str, Any],
    timestamp_generator: Optional[TimestampGenerator] = None
) -> Event:
    """Create a face event."""
    if timestamp_generator is None:
        timestamp_generator = TimestampGenerator()
    return Event(
        session_id=session_id,
        timestamp=timestamp_generator.now(),
        modality=ModalityType.FACE,
        event_type=event_type,
        data=data,
    )