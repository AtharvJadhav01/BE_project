"""
Session Manager - coordinates session lifecycle and modalities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.config import AppConfig, load_config
from core.events import (
    Event,
    EventType,
    ModalityType,
    create_system_event,
)
from core.interfaces import ModalityInterface, ModalityStatus
from core.logging import get_logger, setup_logging
from core.timestamps import TimestampGenerator, get_timestamp_generator
from processing.synchronization import EventSynchronizer, get_synchronization_manager
from storage import SessionStore

from .metadata import SessionMetadata, ModalityStatusInfo
from .state import SessionState, SessionStateMachine


class SessionManager:
    """
    Manages the lifecycle of a capture session.

    Coordinates modality modules, handles session state transitions,
    manages timestamps, and emits system events.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        config_path: Optional[Path] = None,
        application_version: str = "0.1.0",
    ):
        """
        Initialize the Session Manager.

        Args:
            config: Optional pre-loaded configuration
            config_path: Path to config file (used if config not provided)
            application_version: Application version string
        """
        self.logger = get_logger(__name__)
        self.application_version = application_version

        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Setup logging
        setup_logging(self.config.logging)

        # Session state
        self._state_machine = SessionStateMachine()
        self._metadata: Optional[SessionMetadata] = None
        self._timestamp_generator = TimestampGenerator()
        self._modalities: dict[str, ModalityInterface] = {}
        self._event_callbacks: list[Callable[[Event], None]] = []

        # Session store for metadata persistence
        self._session_store = SessionStore(self.config.to_dict() if hasattr(self.config, 'to_dict') else {})

        # Synchronization
        self._synchronization_manager = get_synchronization_manager()
        self._synchronizer: Optional[EventSynchronizer] = None

        # Session directory
        self._session_dir: Optional[Path] = None

    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state_machine.state

    @property
    def metadata(self) -> Optional[SessionMetadata]:
        """Get current session metadata."""
        return self._metadata

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._metadata.session_id if self._metadata else None

    @property
    def timestamp_generator(self) -> TimestampGenerator:
        """Get the timestamp generator."""
        return self._timestamp_generator

    @property
    def modalities(self) -> dict[str, ModalityInterface]:
        """Get registered modalities."""
        return self._modalities.copy()

    def register_modality(self, modality: ModalityInterface) -> None:
        """
        Register a modality module.

        Args:
            modality: Modality instance implementing ModalityInterface
        """
        self._modalities[modality.name] = modality
        self.logger.info("modality_registered", modality=modality.name)

    def unregister_modality(self, name: str) -> None:
        """Unregister a modality module."""
        if name in self._modalities:
            del self._modalities[name]
            self.logger.info("modality_unregistered", modality=name)

    def add_event_callback(self, callback: Callable[[Event], None]) -> None:
        """Add a callback to receive system events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[Event], None]) -> None:
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def _emit_event(self, event: Event) -> None:
        """Emit event to synchronizer for unified event stream."""
        # Also emit to registered callbacks (for backward compatibility)
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error("event_callback_error", error=str(e), callback=callback.__name__)
        
        # Send to synchronizer for unified event stream
        if self._synchronizer:
            self._synchronizer.add_event(event)

    def _create_session_directory(self, session_id: str) -> Path:
        """Create session directory structure."""
        base_dir = Path(self.config.storage.output_directory)
        session_dir = base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "audio").mkdir(exist_ok=True)
        return session_dir

    def start_session(self) -> bool:
        """
        Start a new capture session.

        Returns:
            True if session started successfully, False otherwise
        """
        if self._state_machine.state != SessionState.READY:
            self.logger.warning("session_start_failed", reason="invalid_state", current_state=self._state_machine.state.value)
            return False

        # Generate session timestamps
        start_time_monotonic = self._timestamp_generator._session_start_monotonic or 0.0
        start_time_wallclock = self._timestamp_generator._session_start_wallclock or 0.0

        if start_time_monotonic == 0.0:
            self._timestamp_generator.start_session()
            start_time_monotonic = self._timestamp_generator._session_start_monotonic or 0.0
            start_time_wallclock = self._timestamp_generator._session_start_wallclock or 0.0

        start_wallclock_iso = datetime.fromtimestamp(start_time_wallclock, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        # Determine enabled modalities
        enabled_modalities = []
        for name, modality in self._modalities.items():
            if modality.config.get("enabled", True):
                enabled_modalities.append(name)

        # Create session metadata using SessionStore
        config_snapshot = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        self._metadata = self._session_store.create_session(
            session_id=None,  # Let SessionStore generate UUID
            timestamp_generator=self._timestamp_generator,
            application_version=self.application_version,
            enabled_modalities=enabled_modalities,
            config_snapshot=config_snapshot,
            start_time_monotonic=start_time_monotonic,
            start_time_wallclock=start_wallclock_iso,
        )

        self._session_dir = self._session_store.get_session_dir()
        self._timestamp_generator = self._session_store.get_timestamp_generator()

        # Create and start synchronizer for this session
        self._synchronizer = self._synchronization_manager.create_synchronizer(
            session_id=self._metadata.session_id,
            session_dir=self._session_dir,
            timestamp_generator=self._timestamp_generator
        )

        # Transition state
        if not self._state_machine.start_recording():
            self.logger.error("session_state_transition_failed", target="recording")
            return False

        # Emit session start event
        event = create_system_event(
            session_id=self._metadata.session_id,
            event_type=EventType.SESSION_START,
            data={
                "config_snapshot": config_snapshot,
                "enabled_modalities": enabled_modalities,
            },
            timestamp_generator=self._timestamp_generator,
        )
        self._emit_event(event)

        # Start enabled modalities
        for name in enabled_modalities:
            if name in self._modalities:
                modality = self._modalities[name]
                try:
                    success = modality.start(self._metadata.session_id, self._session_dir)
                    if success:
                        self._session_store.update_modality_status(name, "running")
                        self.logger.info("modality_started", modality=name)
                    else:
                        self._session_store.update_modality_status(name, "error", f"Failed to start {name}")
                        self.logger.error("modality_start_failed", modality=name)
                except Exception as e:
                    self._session_store.update_modality_status(name, "error", str(e))
                    self.logger.error("modality_start_exception", modality=name, error=str(e))

        self.logger.info("session_started", session_id=self._metadata.session_id)
        return True

    def stop_session(self) -> bool:
        """
        Stop the current capture session.

        Returns:
            True if session stopped successfully, False otherwise
        """
        if self._state_machine.state not in (SessionState.RECORDING, SessionState.PAUSED):
            self.logger.warning("session_stop_failed", reason="invalid_state", current_state=self._state_machine.state.value)
            return False

        if not self._metadata:
            self.logger.error("session_stop_failed", reason="no_active_session")
            return False

        # Stop all running modalities
        for name, modality in self._modalities.items():
            if modality.status == ModalityStatus.RUNNING:
                try:
                    success = modality.stop()
                    if success:
                        self._session_store.update_modality_status(name, "completed")
                        self.logger.info("modality_stopped", modality=name)
                    else:
                        self._session_store.update_modality_status(name, "error", f"Failed to stop {name}")
                        self.logger.error("modality_stop_failed", modality=name)
                except Exception as e:
                    self._session_store.update_modality_status(name, "error", str(e))
                    self.logger.error("modality_stop_exception", modality=name, error=str(e))

        # Transition to stopped state
        self._state_machine.stop()

        # Record end timestamps - use the timestamp generator to get consistent times
        end_timestamp = self._timestamp_generator.now()
        end_time_monotonic = end_timestamp.monotonic
        end_time_wallclock = end_timestamp.wallclock

        # Mark session as completed using SessionStore
        self._session_store.mark_session_completed(end_time_monotonic, end_time_wallclock)

        # Emit session end event
        event = create_system_event(
            session_id=self._metadata.session_id,
            event_type=EventType.SESSION_END,
            data={
                "duration_ms": self._metadata.duration_ms,
                "modality_status": {
                    name: info.status for name, info in self._metadata.modality_status.items()
                },
            },
            timestamp_generator=self._timestamp_generator,
        )
        self._emit_event(event)

        self.logger.info("session_stopped", session_id=self._metadata.session_id, duration_ms=self._metadata.duration_ms)
        
        # Stop synchronizer and flush remaining events
        if self._synchronizer:
            self._synchronizer.stop()
            self._synchronization_manager.remove_synchronizer(self._metadata.session_id)
            self._synchronizer = None
        
        return True

    def pause_session(self) -> bool:
        """
        Pause the current capture session.

        Returns:
            True if session paused successfully, False otherwise
        """
        if self._state_machine.state != SessionState.RECORDING:
            self.logger.warning("session_pause_failed", reason="invalid_state", current_state=self._state_machine.state.value)
            return False

        if not self._metadata:
            self.logger.error("session_pause_failed", reason="no_active_session")
            return False

        # Pause all running modalities
        paused_modalities = []
        for name, modality in self._modalities.items():
            if modality.status == ModalityStatus.RUNNING:
                try:
                    success = modality.pause()
                    if success:
                        self._session_store.update_modality_status(name, "paused")
                        paused_modalities.append(name)
                        self.logger.info("modality_paused", modality=name)
                    else:
                        self.logger.warning("modality_pause_failed", modality=name)
                except Exception as e:
                    self.logger.error("modality_pause_exception", modality=name, error=str(e))

        # Transition state
        if not self._state_machine.pause():
            self.logger.error("session_state_transition_failed", target="paused")
            return False

        # Emit pause event
        event = create_system_event(
            session_id=self._metadata.session_id,
            event_type=EventType.SESSION_PAUSE,
            data={"paused_modalities": paused_modalities},
            timestamp_generator=self._timestamp_generator,
        )
        self._emit_event(event)

        self.logger.info("session_paused", session_id=self._metadata.session_id)
        return True

    def resume_session(self) -> bool:
        """
        Resume a paused capture session.

        Returns:
            True if session resumed successfully, False otherwise
        """
        if self._state_machine.state != SessionState.PAUSED:
            self.logger.warning("session_resume_failed", reason="invalid_state", current_state=self._state_machine.state.value)
            return False

        if not self._metadata:
            self.logger.error("session_resume_failed", reason="no_active_session")
            return False

        # Resume paused modalities
        resumed_modalities = []
        for name, modality in self._modalities.items():
            if modality.status == ModalityStatus.PAUSED:
                try:
                    success = modality.resume()
                    if success:
                        self._session_store.update_modality_status(name, "running")
                        resumed_modalities.append(name)
                        self.logger.info("modality_resumed", modality=name)
                    else:
                        self.logger.warning("modality_resume_failed", modality=name)
                except Exception as e:
                    self.logger.error("modality_resume_exception", modality=name, error=str(e))

        # Transition state
        if not self._state_machine.resume():
            self.logger.error("session_state_transition_failed", target="recording")
            return False

        # Emit resume event
        event = create_system_event(
            session_id=self._metadata.session_id,
            event_type=EventType.SESSION_RESUME,
            data={"resumed_modalities": resumed_modalities},
            timestamp_generator=self._timestamp_generator,
        )
        self._emit_event(event)

        self.logger.info("session_resumed", session_id=self._metadata.session_id)
        return True

    def get_session_status(self) -> dict[str, Any]:
        """
        Get comprehensive session status.

        Returns:
            Dictionary with session status information
        """
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "metadata": self._metadata.to_dict() if self._metadata else None,
            "modalities": {
                name: modality.get_info().__dict__
                for name, modality in self._modalities.items()
            },
        }

    def cleanup(self) -> None:
        """Clean up resources and reset state."""
        # Stop synchronizer if active
        if self._synchronizer:
            self._synchronizer.stop()
            if self._metadata and self._metadata.session_id:
                self._synchronization_manager.remove_synchronizer(self._metadata.session_id)
            self._synchronizer = None
        
        self._session_store.clear_active_session()
        
        self._modalities.clear()
        self._event_callbacks.clear()
        self._timestamp_generator.reset()
        self._state_machine = SessionStateMachine()
        self._metadata = None
        self._session_dir = None
        self.logger.info("session_manager_cleaned_up")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(
    config: Optional[AppConfig] = None,
    config_path: Optional[Path] = None,
    application_version: str = "0.1.0",
) -> SessionManager:
    """
    Get or create the global session manager instance.

    Args:
        config: Optional pre-loaded configuration
        config_path: Path to config file
        application_version: Application version string

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(config, config_path, application_version)
    return _session_manager


def reset_session_manager() -> None:
    """Reset the global session manager instance."""
    global _session_manager
    if _session_manager is not None:
        _session_manager.cleanup()
        _session_manager = None