"""
Session module for the Multimodal Human State Capture Application.

Provides session metadata, state management, and Session Manager.
"""

from .manager import (
    SessionManager,
    get_session_manager,
    reset_session_manager,
)

from .metadata import (
    ModalityStatusInfo,
    SessionMetadata,
)

from .state import (
    SessionState,
    SessionStateMachine,
)

__all__ = [
    "SessionManager",
    "get_session_manager",
    "reset_session_manager",
    "ModalityStatusInfo",
    "SessionMetadata",
    "SessionState",
    "SessionStateMachine",
]