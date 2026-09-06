"""
Core module for the Multimodal Human State Capture Application.

Provides configuration, logging, timestamps, interfaces, and events.
"""

from .config import (
    AppConfig,
    AudioConfig,
    CameraConfig,
    FaceConfig,
    KeyboardConfig,
    LoggingConfig,
    StorageConfig,
    UIConfig,
    get_default_config,
    load_config,
)

from .events import (
    Event,
    EventType,
    ModalityType,
    create_audio_event,
    create_face_event,
    create_keyboard_event,
    create_system_event,
)

from .interfaces import (
    AudioModalityInterface,
    FaceModalityInterface,
    KeyboardModalityInterface,
    ModalityInfo,
    ModalityInterface,
    ModalityStatus,
)

from .logging import (
    bind_session_context,
    get_logger,
    setup_logging,
)

from .timestamps import (
    Timestamp,
    TimestampGenerator,
    get_timestamp_generator,
    now,
)

__all__ = [
    # Config
    "AppConfig",
    "AudioConfig",
    "CameraConfig",
    "FaceConfig",
    "KeyboardConfig",
    "LoggingConfig",
    "StorageConfig",
    "UIConfig",
    "load_config",
    "get_default_config",
    # Events
    "Event",
    "EventType",
    "ModalityType",
    "create_audio_event",
    "create_face_event",
    "create_keyboard_event",
    "create_system_event",
    # Interfaces
    "AudioModalityInterface",
    "FaceModalityInterface",
    "KeyboardModalityInterface",
    "ModalityInfo",
    "ModalityInterface",
    "ModalityStatus",
    # Logging
    "bind_session_context",
    "get_logger",
    "setup_logging",
    # Timestamps
    "Timestamp",
    "TimestampGenerator",
    "get_timestamp_generator",
    "now",
]