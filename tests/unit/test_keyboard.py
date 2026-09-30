"""
Unit tests for keyboard capture module.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.events import Event, EventType, ModalityType
from core.timestamps import TimestampGenerator
from keyboard.listener import KeyboardListener
from keyboard.processor import KeyboardProcessor, ProcessedKeyboardEvent
from keyboard.storage import KeyboardStorage
from keyboard.capture import KeyboardCapture
from analysis.keyboard.features import KeyboardFeatureExtractor, KeyboardFeatures


class TestKeyboardFeatureExtractor:
    def test_typing_speed_and_correction_rate_use_key_timestamps(self):
        extractor = KeyboardFeatureExtractor("keyboard", {})

        extractor._process_key_down("a", True, [], 0.0)
        extractor._process_key_down("Backspace", False, [], 60.0)
        result = KeyboardFeatures()
        extractor._compute_features(result, 60.0)

        assert extractor._key_timestamps == [0.0, 60.0]
        assert result.typing_speed_kpm == 2.0
        assert result.correction_rate == 1.0

    def test_consistent_intervals_have_maximum_rhythm_regularity(self):
        extractor = KeyboardFeatureExtractor("keyboard", {})
        extractor._inter_key_intervals = [100.0, 100.0, 100.0]

        result = KeyboardFeatures()
        extractor._compute_features(result, 1.0)

        assert result.typing_rhythm_regularity == 1.0

    def test_rhythm_regularity_changes_with_interval_variation(self):
        extractor = KeyboardFeatureExtractor("keyboard", {})
        extractor._inter_key_intervals = [100.0, 200.0, 300.0]

        result = KeyboardFeatures()
        extractor._compute_features(result, 1.0)

        assert 0.0 < result.typing_rhythm_regularity < 1.0


class TestKeyboardListener:
    """Tests for KeyboardListener class."""

    @pytest.fixture
    def timestamp_generator(self):
        """Create a timestamp generator."""
        gen = TimestampGenerator()
        gen.start_session()
        return gen

    @pytest.fixture
    def mock_callback(self):
        """Create a mock event callback."""
        return Mock()

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "enabled": True,
            "capture_modifiers": True,
            "capture_context": True,
            "exclude_sensitive": True,
        }

    def test_listener_creation(self, timestamp_generator, mock_callback, config):
        """Test listener creation."""
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        assert listener.session_id == "test-session"
        assert not listener.is_running
        assert not listener.is_paused

    def test_normalize_printable_key(self, timestamp_generator, mock_callback, config):
        """Test normalization of printable keys."""
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        # Mock a printable key
        from pynput.keyboard import KeyCode
        key = KeyCode(char='a')
        key_name, is_printable, key_code = listener._normalize_key(key)
        
        assert key_name == 'a'
        assert is_printable is True
        assert key_code == 65  # ASCII for 'A'

    def test_normalize_special_key(self, timestamp_generator, mock_callback, config):
        """Test normalization of special keys."""
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        from pynput.keyboard import Key
        key_name, is_printable, key_code = listener._normalize_key(Key.enter)
        
        assert key_name == 'Enter'
        assert is_printable is False
        assert key_code == 13

    def test_normalize_modifier_key(self, timestamp_generator, mock_callback, config):
        """Test normalization of modifier keys."""
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        from pynput.keyboard import Key
        key_name, is_printable, key_code = listener._normalize_key(Key.shift)
        
        assert key_name in ('Shift_L', 'Shift_R')
        assert is_printable is False
        assert key_code == 16

    def test_start_stop(self, timestamp_generator, mock_callback, config):
        """Test listener start/stop."""
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        # Start listener
        result = listener.start()
        assert result is True
        assert listener.is_running
        
        # Stop listener
        result = listener.stop()
        assert result is True
        assert not listener.is_running

    def test_pause_resume(self, timestamp_generator, mock_callback, config):
        """Test listener pause/resume."""
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        listener.start()
        
        # Pause
        result = listener.pause()
        assert result is True
        assert listener.is_paused
        
        # Resume
        result = listener.resume()
        assert result is True
        assert not listener.is_paused
        
        listener.stop()

    def test_sensitive_context_detection(self, timestamp_generator, mock_callback, config):
        """Test sensitive context detection."""
        config["exclude_sensitive"] = True
        listener = KeyboardListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        # Set sensitive window title
        listener._current_window_title = "Login - Password Required"
        
        assert listener._is_sensitive_context() is True
        
        # Non-sensitive window
        listener._current_window_title = "Notepad"
        assert listener._is_sensitive_context() is False


class TestKeyboardProcessor:
    """Tests for KeyboardProcessor class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "enabled": True,
            "capture_modifiers": True,
            "capture_context": True,
            "exclude_sensitive": True,
        }

    @pytest.fixture
    def processor(self, config):
        """Create a processor instance."""
        return KeyboardProcessor(config)

    @pytest.fixture
    def mock_event(self, config):
        """Create a mock keyboard event."""
        gen = TimestampGenerator()
        gen.start_session()
        timestamp = gen.now()
        
        return Event(
            session_id="test-session",
            timestamp=timestamp,
            modality=ModalityType.KEYBOARD,
            event_type=EventType.KEY_DOWN,
            data={
                "event_type": "key_down",
                "key": "a",
                "key_code": 65,
                "modifiers": [],
                "is_printable": True,
                "is_sensitive_context": False,
            }
        )

    def test_processor_start_stop(self, processor):
        """Test processor start/stop."""
        assert processor.start() is True
        assert processor._processing is True
        
        assert processor.stop() is True
        assert processor._processing is False

    def test_process_event(self, processor, mock_event):
        """Test processing a keyboard event."""
        processor.start()
        
        # Set up mock storage callback
        storage_callback = Mock()
        processor.set_storage_callback(storage_callback)
        
        # Process event
        result = processor.process_event(mock_event)
        assert result is True
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify callback was called
        storage_callback.assert_called_once()
        
        processor.stop()

    def test_process_event_with_unified_callback(self, processor, mock_event):
        """Test processing with unified event callback."""
        processor.start()
        
        unified_callback = Mock()
        processor.set_unified_event_callback(unified_callback)
        
        result = processor.process_event(mock_event)
        assert result is True
        
        time.sleep(0.1)
        
        unified_callback.assert_called_once_with(mock_event)
        
        processor.stop()

    def test_stats(self, processor, mock_event):
        """Test processor statistics."""
        processor.start()
        processor.process_event(mock_event)
        time.sleep(0.1)
        
        stats = processor.get_stats()
        assert stats["events_processed"] == 1
        assert stats["events_dropped"] == 0
        
        processor.stop()


class TestKeyboardStorage:
    """Tests for KeyboardStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {}

    @pytest.fixture
    def storage(self, temp_dir, config):
        """Create storage instance."""
        return KeyboardStorage(temp_dir / "test-session", config)

    @pytest.fixture
    def sample_event(self):
        """Create a sample processed event."""
        gen = TimestampGenerator()
        gen.start_session()
        timestamp = gen.now()
        
        return ProcessedKeyboardEvent(
            event_id="evt_001",
            timestamp_monotonic=timestamp.monotonic,
            timestamp_wallclock=timestamp.wallclock,
            event_type="key_down",
            key="a",
            key_code=65,
            modifiers=[],
            duration_ms=None,
            is_printable=True,
            is_sensitive_context=False,
            window_title="Notepad",
            process_name="notepad.exe",
        )

    def test_storage_start_stop(self, storage):
        """Test storage start/stop."""
        assert storage.start() is True
        assert storage._writing is True
        
        assert storage.stop() is True
        assert storage._writing is False

    def test_write_event(self, storage, sample_event):
        """Test writing an event."""
        storage.start()
        
        result = storage.write_event(sample_event)
        assert result is True
        
        # Wait for write
        time.sleep(0.1)
        
        # Check file was created
        file_path = storage._file_path
        assert file_path.exists()
        
        # Read and verify content
        import orjson
        with open(file_path, 'rb') as f:
            line = f.readline()
            data = orjson.loads(line)
            assert data["key"] == "a"
            assert data["event_type"] == "key_down"
        
        storage.stop()

    def test_multiple_events(self, storage, sample_event):
        """Test writing multiple events."""
        storage.start()
        
        # Write multiple events
        for i in range(5):
            event = ProcessedKeyboardEvent(
                event_id=f"evt_{i:03d}",
                timestamp_monotonic=sample_event.timestamp_monotonic + i * 10,
                timestamp_wallclock=sample_event.timestamp_wallclock,
                event_type="key_down" if i % 2 == 0 else "key_up",
                key="a" if i % 2 == 0 else "b",
                key_code=65 if i % 2 == 0 else 66,
                modifiers=[],
                duration_ms=50.0 if i % 2 == 1 else None,
                is_printable=True,
                is_sensitive_context=False,
                window_title="Notepad",
                process_name="notepad.exe",
            )
            storage.write_event(event)
        
        time.sleep(0.2)
        
        # Verify all events written
        import orjson
        with open(storage._file_path, 'rb') as f:
            lines = f.readlines()
            assert len(lines) == 5
        
        stats = storage.get_stats()
        assert stats["events_written"] == 5
        
        storage.stop()


class TestKeyboardCapture:
    """Tests for KeyboardCapture (modality implementation)."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "enabled": True,
            "capture_modifiers": True,
            "capture_context": True,
            "exclude_sensitive": True,
            "storage": {
                "output_directory": "data/sessions"
            }
        }

    @pytest.fixture
    def capture(self, config):
        """Create keyboard capture instance."""
        return KeyboardCapture(config)

    def test_capture_creation(self, capture):
        """Test capture creation."""
        assert capture.name == "keyboard"
        assert capture.status.value == "stopped"

    def test_capture_start_stop(self, capture, temp_dir):
        """Test capture start/stop."""
        config = capture.config.copy()
        config["storage"]["output_directory"] = str(temp_dir)
        
        # Mock unified event callback
        unified_callback = Mock()
        capture.set_unified_event_callback(unified_callback)
        
        # Start capture
        result = capture.start("test-session")
        assert result is True
        assert capture.status.value == "running"
        
        # Stop capture
        result = capture.stop()
        assert result is True
        assert capture.status.value == "stopped"

    def test_capture_pause_resume(self, capture, temp_dir):
        """Test capture pause/resume."""
        config = capture.config.copy()
        config["storage"]["output_directory"] = str(temp_dir)
        
        unified_callback = Mock()
        capture.set_unified_event_callback(unified_callback)
        
        capture.start("test-session")
        
        # Pause
        result = capture.pause()
        assert result is True
        assert capture.status.value == "paused"
        
        # Resume
        result = capture.resume()
        assert result is True
        assert capture.status.value == "running"
        
        capture.stop()

    def test_get_stats(self, capture, temp_dir):
        """Test getting capture statistics."""
        config = capture.config.copy()
        config["storage"]["output_directory"] = str(temp_dir)
        
        unified_callback = Mock()
        capture.set_unified_event_callback(unified_callback)
        
        capture.start("test-session")
        
        stats = capture.get_stats()
        assert "status" in stats
        assert "processor" in stats
        assert "storage" in stats
        assert "listener" in stats
        
        capture.stop()


class TestKeyboardIntegration:
    """Integration tests for keyboard capture with session manager."""

    def test_keyboard_with_session_manager(self, temp_dir):
        """Test keyboard capture integrated with session manager."""
        from core.config import get_default_config
        from session.manager import SessionManager
        
        config = get_default_config()
        config.storage.output_directory = str(temp_dir)
        config.keyboard.enabled = True
        config.audio.enabled = False
        config.camera.enabled = False
        
        manager = SessionManager(config=config, application_version="test-1.0.0")
        
        # Register keyboard capture
        from keyboard import KeyboardCapture
        keyboard_capture = KeyboardCapture(config.keyboard.__dict__)
        keyboard_capture.set_unified_event_callback(
            lambda event: manager._emit_event(event)
        )
        manager.register_modality(keyboard_capture)
        
        # Start session
        result = manager.start_session()
        assert result is True
        assert manager.state.value == "recording"
        
        # Wait a bit for any async operations
        time.sleep(0.1)
        
        # Stop session
        result = manager.stop_session()
        assert result is True
        assert manager.state.value == "stopped"
        
        # Check keystrokes file was created
        session_dir = temp_dir / manager.session_id
        keystrokes_file = session_dir / "keystrokes.jsonl"
        # File should exist even if no keys pressed
        assert keystrokes_file.exists()


# Need temp_dir fixture
@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)