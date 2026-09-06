"""
Processing module for the Multimodal Human State Capture Application.

Provides synchronization and event processing utilities.
"""

from .synchronization import (
    EventSynchronizer,
    SynchronizationManager,
    get_synchronization_manager,
    reset_synchronization_manager,
)

__all__ = [
    "EventSynchronizer",
    "SynchronizationManager",
    "get_synchronization_manager",
    "reset_synchronization_manager",
]