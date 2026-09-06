"""
Interfaces and contracts for modality modules.

Defines the common interface that all modality capture modules must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ModalityStatus(Enum):
    """Status of a modality module."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ModalityInfo:
    """Information about a modality's current state."""
    name: str
    status: ModalityStatus
    enabled: bool
    error_message: Optional[str] = None
    last_error: Optional[Exception] = None


class ModalityInterface(ABC):
    """
    Abstract base class for all modality capture modules.

    Each modality (audio, keyboard, face) must implement this interface
    to be managed by the Session Manager.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self._status = ModalityStatus.STOPPED
        self._error_message: Optional[str] = None
        self._last_error: Optional[Exception] = None

    @property
    def status(self) -> ModalityStatus:
        """Get current modality status."""
        return self._status

    @property
    def error_message(self) -> Optional[str]:
        """Get last error message if any."""
        return self._error_message

    @property
    def last_error(self) -> Optional[Exception]:
        """Get last exception if any."""
        return self._last_error

    def get_info(self) -> ModalityInfo:
        """Get current modality information."""
        return ModalityInfo(
            name=self.name,
            status=self._status,
            enabled=self.config.get("enabled", True),
            error_message=self._error_message,
            last_error=self._last_error,
        )

    @abstractmethod
    def start(self, session_id: str, session_dir: Optional[Path] = None) -> bool:
        """
        Start the modality capture.

        Args:
            session_id: Unique session identifier
            session_dir: Optional session directory path

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the modality capture.

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def pause(self) -> bool:
        """
        Pause the modality capture.

        Returns:
            True if paused successfully, False otherwise
        """
        pass

    @abstractmethod
    def resume(self) -> bool:
        """
        Resume the modality capture after pause.

        Returns:
            True if resumed successfully, False otherwise
        """
        pass

    def _set_status(self, status: ModalityStatus) -> None:
        """Internal method to update status."""
        self._status = status

    def _set_error(self, error: Exception, message: str) -> None:
        """Internal method to set error state."""
        self._last_error = error
        self._error_message = message
        self._status = ModalityStatus.ERROR


class AudioModalityInterface(ModalityInterface):
    """Interface for audio capture modality."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("audio", config)


class KeyboardModalityInterface(ModalityInterface):
    """Interface for keyboard capture modality."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("keyboard", config)


class FaceModalityInterface(ModalityInterface):
    """Interface for facial capture modality."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("face", config)