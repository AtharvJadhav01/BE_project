"""
Unit tests for Session Manager.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config import AppConfig, get_default_config
from core.interfaces import ModalityInterface, ModalityStatus
from session.manager import SessionManager
from session.metadata import SessionMetadata
from session.state import SessionState, SessionStateMachine


class DummyModality(ModalityInterface):
    """Dummy modality for testing."""
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config or {"enabled": True})
        self.start_called = False
        self.stop_called = False
        self.pause_called = False
        self.resume_called = False
        self.should_fail = False

    def start(self, session_id: str, session_dir: Path = None) -> bool:
        self.start_called = True
        if self.should_fail:
            self._set_error(Exception("Simulated failure"), "Simulated failure")
            return False
        self._set_status(ModalityStatus.RUNNING)
        return True

    def stop(self) -> bool:
        self.stop_called = True
        if self.should_fail:
            return False
        self._set_status(ModalityStatus.STOPPED)
        return True

    def pause(self) -> bool:
        self.pause_called = True
        if self.should_fail:
            return False
        if self._status == ModalityStatus.RUNNING:
            self._set_status(ModalityStatus.PAUSED)
            return True
        return False

    def resume(self) -> bool:
        self.resume_called = True
        if self.should_fail:
            return False
        if self._status == ModalityStatus.PAUSED:
            self._set_status(ModalityStatus.RUNNING)
            return True
        return False


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test output."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = get_default_config()
        config.storage.output_directory = str(temp_dir / "sessions")
        return config

    @pytest.fixture
    def session_manager(self, config):
        """Create a SessionManager instance with dummy modalities."""
        manager = SessionManager(config=config, application_version="test-1.0.0")

        # Register dummy modalities
        audio = DummyModality("audio", {"enabled": True})
        keyboard = DummyModality("keyboard", {"enabled": True})
        face = DummyModality("face", {"enabled": True})

        manager.register_modality(audio)
        manager.register_modality(keyboard)
        manager.register_modality(face)

        yield manager
        
        # Cleanup after test
        manager.cleanup()

    def test_initial_state(self, session_manager):
        """Test initial session manager state."""
        assert session_manager.state == SessionState.READY
        assert session_manager.metadata is None
        assert session_manager.session_id is None

    def test_register_modality(self, session_manager):
        """Test modality registration."""
        assert "audio" in session_manager.modalities
        assert "keyboard" in session_manager.modalities
        assert "face" in session_manager.modalities

    def test_unregister_modality(self, session_manager):
        """Test modality unregistration."""
        session_manager.unregister_modality("audio")
        assert "audio" not in session_manager.modalities
        assert "keyboard" in session_manager.modalities

    def test_start_session(self, session_manager):
        """Test starting a session."""
        result = session_manager.start_session()
        assert result is True
        assert session_manager.state == SessionState.RECORDING
        assert session_manager.metadata is not None
        assert session_manager.session_id is not None

        # Check that modalities were started
        assert session_manager.modalities["audio"].start_called
        assert session_manager.modalities["keyboard"].start_called
        assert session_manager.modalities["face"].start_called

        # Check modality statuses in metadata
        for name in ["audio", "keyboard", "face"]:
            assert session_manager.metadata.modality_status[name].status == "running"

    def test_start_session_invalid_state(self, session_manager):
        """Test starting session from invalid state."""
        session_manager.start_session()
        # Try to start again
        result = session_manager.start_session()
        assert result is False
        assert session_manager.state == SessionState.RECORDING

    def test_stop_session(self, session_manager):
        """Test stopping a session."""
        session_manager.start_session()
        # Small delay to ensure measurable duration
        import time
        time.sleep(0.01)
        result = session_manager.stop_session()

        assert result is True
        assert session_manager.state == SessionState.STOPPED
        assert session_manager.metadata.status == "completed"
        assert session_manager.metadata.end_time_wallclock is not None
        assert session_manager.metadata.duration_ms is not None
        assert session_manager.metadata.duration_ms > 0

        # Check that modalities were stopped
        assert session_manager.modalities["audio"].stop_called
        assert session_manager.modalities["keyboard"].stop_called
        assert session_manager.modalities["face"].stop_called

    def test_stop_session_invalid_state(self, session_manager):
        """Test stopping session from invalid state."""
        result = session_manager.stop_session()
        assert result is False
        assert session_manager.state == SessionState.READY

    def test_pause_session(self, session_manager):
        """Test pausing a session."""
        session_manager.start_session()
        result = session_manager.pause_session()

        assert result is True
        assert session_manager.state == SessionState.PAUSED

        # Check that modalities were paused
        assert session_manager.modalities["audio"].pause_called
        assert session_manager.modalities["keyboard"].pause_called
        assert session_manager.modalities["face"].pause_called

    def test_pause_session_invalid_state(self, session_manager):
        """Test pausing session from invalid state."""
        result = session_manager.pause_session()
        assert result is False
        assert session_manager.state == SessionState.READY

    def test_resume_session(self, session_manager):
        """Test resuming a session."""
        session_manager.start_session()
        session_manager.pause_session()
        result = session_manager.resume_session()

        assert result is True
        assert session_manager.state == SessionState.RECORDING

        # Check that modalities were resumed
        assert session_manager.modalities["audio"].resume_called
        assert session_manager.modalities["keyboard"].resume_called
        assert session_manager.modalities["face"].resume_called

    def test_resume_session_invalid_state(self, session_manager):
        """Test resuming session from invalid state."""
        result = session_manager.resume_session()
        assert result is False
        assert session_manager.state == SessionState.READY

    def test_modality_failure_handling(self, session_manager):
        """Test handling of modality failures."""
        # Make audio modality fail
        session_manager.modalities["audio"].should_fail = True

        result = session_manager.start_session()
        assert result is True

        # Audio should be marked as error
        assert session_manager.metadata.modality_status["audio"].status == "error"
        assert session_manager.metadata.error_count == 1

    def test_session_status(self, session_manager):
        """Test getting session status."""
        status = session_manager.get_session_status()
        assert status["session_id"] is None
        assert status["state"] == "ready"

        session_manager.start_session()
        status = session_manager.get_session_status()
        assert status["session_id"] is not None
        assert status["state"] == "recording"
        assert "modalities" in status

    def test_cleanup(self, session_manager):
        """Test cleanup."""
        session_manager.start_session()
        session_manager.cleanup()

        assert session_manager.state == SessionState.READY
        assert session_manager.metadata is None
        assert session_manager.session_id is None
        assert len(session_manager.modalities) == 0


class TestSessionMetadata:
    """Tests for SessionMetadata class."""

    def test_create_new(self):
        """Test creating new session metadata."""
        metadata = SessionMetadata.create_new(
            application_version="1.0.0",
            enabled_modalities=["audio", "keyboard"],
            config_snapshot={"audio": {"enabled": True}},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z",
        )

        assert metadata.session_id is not None
        assert metadata.application_version == "1.0.0"
        assert metadata.start_time_wallclock == "2024-01-01T00:00:00.000Z"
        assert metadata.status == "recording"
        assert "audio" in metadata.enabled_modalities
        assert "keyboard" in metadata.enabled_modalities

    def test_mark_completed(self):
        """Test marking session as completed."""
        metadata = SessionMetadata.create_new(
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z",
        )

        metadata.mark_completed(10.5, "2024-01-01T00:00:10.500Z")

        assert metadata.end_time_monotonic == 10.5
        assert metadata.end_time_wallclock == "2024-01-01T00:00:10.500Z"
        assert metadata.duration_ms == 10500
        assert metadata.status == "completed"

    def test_update_modality_status(self):
        """Test updating modality status."""
        metadata = SessionMetadata.create_new(
            application_version="1.0.0",
            enabled_modalities=["audio", "keyboard"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z",
        )

        metadata.update_modality_status("audio", "error", "Microphone not found")

        assert metadata.modality_status["audio"].status == "error"
        assert metadata.modality_status["audio"].error_message == "Microphone not found"
        assert metadata.error_count == 1

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        metadata = SessionMetadata.create_new(
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={"test": "value"},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z",
        )

        metadata.update_modality_status("audio", "running")

        data = metadata.to_dict()
        restored = SessionMetadata.from_dict(data)

        assert restored.session_id == metadata.session_id
        assert restored.application_version == metadata.application_version
        assert restored.enabled_modalities == metadata.enabled_modalities
        assert restored.config_snapshot == metadata.config_snapshot
        assert restored.modality_status["audio"].status == "running"


class TestSessionStateMachine:
    """Tests for SessionStateMachine class."""

    def test_initial_state(self):
        """Test initial state."""
        machine = SessionStateMachine()
        assert machine.state == SessionState.READY

    def test_valid_transitions(self):
        """Test valid state transitions."""
        machine = SessionStateMachine()

        assert machine.start_recording()
        assert machine.state == SessionState.RECORDING

        assert machine.pause()
        assert machine.state == SessionState.PAUSED

        assert machine.resume()
        assert machine.state == SessionState.RECORDING

        assert machine.stop()
        assert machine.state == SessionState.STOPPED

        assert machine.complete()
        assert machine.state == SessionState.COMPLETED

    def test_invalid_transitions(self):
        """Test invalid state transitions."""
        machine = SessionStateMachine()

        # Can't pause from READY
        assert not machine.pause()
        assert machine.state == SessionState.READY

        # Can't resume from READY
        assert not machine.resume()
        assert machine.state == SessionState.READY

        # Can't stop from READY
        assert not machine.stop()
        assert machine.state == SessionState.READY

    def test_can_transition(self):
        """Test can_transition method."""
        machine = SessionStateMachine()

        assert machine.can_transition(SessionState.RECORDING)
        assert not machine.can_transition(SessionState.PAUSED)
        assert not machine.can_transition(SessionState.COMPLETED)

        machine.start_recording()
        assert machine.can_transition(SessionState.PAUSED)
        assert machine.can_transition(SessionState.STOPPED)
        assert not machine.can_transition(SessionState.READY)

    def test_reset(self):
        """Test reset."""
        machine = SessionStateMachine()
        machine.start_recording()
        machine.reset()

        assert machine.state == SessionState.READY