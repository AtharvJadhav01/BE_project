"""
Unit tests for session storage module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.timestamps import TimestampGenerator
from storage import SessionStore, SessionLoader


class TestSessionStore:
    """Tests for SessionStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test output."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            "storage": {
                "output_directory": str(temp_dir / "sessions")
            }
        }

    @pytest.fixture
    def session_store(self, config):
        """Create a SessionStore instance."""
        return SessionStore(config)

    def test_store_creation(self, session_store):
        """Test SessionStore creation."""
        assert session_store is not None
        assert session_store.output_directory.exists()

    def test_create_session(self, session_store):
        """Test creating a new session."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id=None,
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio", "keyboard"],
            config_snapshot={"audio": {"sample_rate": 16000}},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        assert metadata is not None
        assert metadata.session_id is not None
        assert metadata.application_version == "1.0.0"
        assert metadata.enabled_modalities == ["audio", "keyboard"]
        assert metadata.status == "recording"
        
        # Check session directory was created
        session_dir = session_store.get_session_dir()
        assert session_dir is not None
        assert session_dir.exists()
        assert (session_dir / "session.json").exists()
        assert (session_dir / "audio").exists()

    def test_session_json_persistence(self, session_store):
        """Test that session.json is written and can be read back."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="test-session-123",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        session_dir = session_store.get_session_dir()
        session_file = session_dir / "session.json"
        
        assert session_file.exists()
        
        # Load and verify
        with open(session_file, "rb") as f:
            import orjson
            data = orjson.loads(f.read())
        
        assert data["session_id"] == "test-session-123"
        assert data["application_version"] == "1.0.0"
        assert data["enabled_modalities"] == ["audio"]
        assert data["status"] == "recording"

    def test_update_modality_status(self, session_store):
        """Test updating modality status."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="test-session",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio", "keyboard"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        # Update audio status
        session_store.update_modality_status("audio", "running")
        session_store.update_modality_status("keyboard", "running")
        
        # Verify in metadata
        assert session_store._active_session.modality_status["audio"].status == "running"
        assert session_store._active_session.modality_status["keyboard"].status == "running"
        
        # Verify in session.json
        session_dir = session_store.get_session_dir()
        session_file = session_dir / "session.json"
        with open(session_file, "rb") as f:
            import orjson
            data = orjson.loads(f.read())
        
        assert data["modality_status"]["audio"]["status"] == "running"
        assert data["modality_status"]["keyboard"]["status"] == "running"

    def test_mark_session_completed(self, session_store):
        """Test marking session as completed."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="test-session",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        # Mark completed
        end_time_monotonic = 5.0
        end_time_wallclock = "2024-01-01T00:00:05.000Z"
        session_store.mark_session_completed(end_time_monotonic, end_time_wallclock)
        
        # Verify
        assert session_store._active_session.status == "completed"
        assert session_store._active_session.end_time_monotonic == 5.0
        assert session_store._active_session.end_time_wallclock == "2024-01-01T00:00:05.000Z"
        assert session_store._active_session.duration_ms == 5000

    def test_mark_session_failed(self, session_store):
        """Test marking session as failed."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="test-session",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        session_store.mark_session_failed(5.0, "2024-01-01T00:00:05.000Z", 3)
        
        assert session_store._active_session.status == "failed"
        assert session_store._active_session.error_count == 3

    def test_save_and_load_session_json(self, session_store, temp_dir):
        """Test saving and loading session.json to/from arbitrary directory."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="test-session",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        # Save to a different directory
        other_dir = temp_dir / "other_sessions" / metadata.session_id
        other_dir.mkdir(parents=True)
        
        result = session_store.save_session_json(other_dir, metadata)
        assert result is True
        
        # Load back
        loaded = session_store.load_session_json(other_dir)
        assert loaded is not None
        assert loaded.session_id == metadata.session_id
        assert loaded.application_version == "1.0.0"

    def test_list_sessions(self, session_store):
        """Test listing sessions."""
        gen = TimestampGenerator()
        gen.start_session()
        
        # Create multiple sessions
        session_store.create_session(
            session_id="session-1",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        # Create another session
        gen2 = TimestampGenerator()
        gen2.start_session()
        session_store.create_session(
            session_id="session-2",
            timestamp_generator=gen2,
            application_version="1.0.0",
            enabled_modalities=["keyboard"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T01:00:00.000Z"
        )
        
        sessions = session_store.list_sessions()
        assert len(sessions) == 2
        assert any(s["session_id"] == "session-1" for s in sessions)
        assert any(s["session_id"] == "session-2" for s in sessions)

    def test_delete_session(self, session_store, temp_dir):
        """Test deleting a session."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="session-to-delete",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        # Delete it
        result = session_store.delete_session(metadata.session_id)
        assert result is True
        
        # Verify it's gone
        session_dir = temp_dir / "sessions" / metadata.session_id
        assert not session_dir.exists()

    def test_get_session_summary(self, session_store):
        """Test getting session summary."""
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = session_store.create_session(
            session_id="test-summary",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio", "keyboard"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        # Mark some modalities as completed
        session_store.update_modality_status("audio", "completed")
        session_store.update_modality_status("keyboard", "running")
        
        summary = session_store.get_session_summary(metadata.session_id)
        
        assert summary is not None
        assert summary["session_id"] == metadata.session_id
        assert summary["enabled_modalities"] == ["audio", "keyboard"]
        assert "modality_status" in summary
        assert summary["modality_status"]["audio"]["status"] == "completed"
        assert summary["modality_status"]["keyboard"]["status"] == "running"


class TestSessionLoader:
    """Tests for SessionLoader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test output."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            "storage": {
                "output_directory": str(temp_dir / "sessions")
            }
        }

    @pytest.fixture
    def loader(self, config):
        """Create a SessionLoader instance."""
        return SessionLoader(config)

    def test_load_session_not_found(self, loader):
        """Test loading non-existent session."""
        result = loader.load_session("non-existent-session")
        assert result is None

    def test_load_session_basic(self, loader, temp_dir):
        """Test loading a session with basic data."""
        from storage import SessionStore
        from core.timestamps import TimestampGenerator
        
        # Create a session using SessionStore
        config = loader.config
        store = SessionStore(config)
        gen = TimestampGenerator()
        gen.start_session()
        
        metadata = store.create_session(
            session_id="test-load-session",
            timestamp_generator=gen,
            application_version="1.0.0",
            enabled_modalities=["audio", "keyboard"],
            config_snapshot={},
            start_time_monotonic=0.0,
            start_time_wallclock="2024-01-01T00:00:00.000Z"
        )
        
        session_dir = temp_dir / "sessions" / metadata.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Write some test data
        import orjson
        with open(session_dir / "keystrokes.jsonl", "wb") as f:
            f.write(orjson.dumps({"event_id": "kbd_1", "key": "a"}) + b"\n")
        
        with open(session_dir / "facial_features.jsonl", "wb") as f:
            f.write(orjson.dumps({"observation_id": "face_1", "face_detected": True}) + b"\n")
        
        with open(session_dir / "events.jsonl", "wb") as f:
            f.write(orjson.dumps({"event_id": "evt_1", "modality": "keyboard"}) + b"\n")
        
        # Load using SessionLoader
        loaded = loader.load_session(metadata.session_id)
        
        assert loaded is not None
        assert loaded["metadata"]["session_id"] == metadata.session_id
        assert len(loaded["keystrokes"]) == 1
        assert len(loaded["facial_features"]) == 1
        assert len(loaded["events"]) == 1


# Need temp_dir fixture
@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)