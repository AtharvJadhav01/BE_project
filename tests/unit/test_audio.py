"""
Unit tests for audio capture module.
"""

import pytest
import tempfile
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.events import Event, EventType, ModalityType
from core.timestamps import TimestampGenerator
from audio.listener import AudioListener
from audio.processor import AudioProcessor, ProcessedAudioChunk
from audio.storage import AudioStorage
from audio.capture import AudioCapture


class TestAudioListener:
    """Tests for AudioListener class."""

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
            "sample_rate": 16000,
            "channels": 1,
            "chunk_duration_ms": 100,
            "device": None,
        }

    def test_listener_creation(self, timestamp_generator, mock_callback, config):
        """Test listener creation."""
        listener = AudioListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        assert listener.session_id == "test-session"
        assert not listener.is_running
        assert not listener.is_paused

    def test_start_stop(self, timestamp_generator, mock_callback, config):
        """Test listener start/stop."""
        listener = AudioListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        with patch('sounddevice.query_devices') as mock_devices:
            mock_devices.return_value = [
                {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
            ]
            
            with patch('sounddevice.InputStream') as mock_stream_class:
                mock_stream = MagicMock()
                mock_stream_class.return_value = mock_stream
                
                result = listener.start()
                assert result is True
                assert listener.is_running
                
                # Stop
                result = listener.stop()
                assert result is True
                assert not listener.is_running

    def test_pause_resume(self, timestamp_generator, mock_callback, config):
        """Test listener pause/resume."""
        listener = AudioListener(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        
        with patch('sounddevice.query_devices') as mock_devices:
            mock_devices.return_value = [
                {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
            ]
            
            with patch('sounddevice.InputStream') as mock_stream_class:
                mock_stream = MagicMock()
                mock_stream_class.return_value = mock_stream
                
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

    def test_check_device_availability(self):
        """Test device availability check."""
        with patch('sounddevice.query_devices') as mock_devices:
            mock_devices.return_value = [
                {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
            ]
            
            assert AudioListener.check_device_availability() is True
            assert AudioListener.check_device_availability(0) is True
            
            # No input devices
            mock_devices.return_value = [
                {'name': 'Output Device', 'max_input_channels': 0, 'default_samplerate': 48000}
            ]
            assert AudioListener.check_device_availability() is False

    def test_get_available_devices(self):
        """Test getting available devices."""
        with patch('sounddevice.query_devices') as mock_devices:
            mock_devices.return_value = [
                {'name': 'Input Device', 'max_input_channels': 2, 'default_samplerate': 48000},
                {'name': 'Output Device', 'max_input_channels': 0, 'default_samplerate': 48000},
            ]
            
            devices = AudioListener.get_available_devices()
            assert len(devices) == 1
            assert devices[0]["name"] == "Input Device"
            assert devices[0]["max_input_channels"] == 2


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "sample_rate": 16000,
            "channels": 1,
        }

    @pytest.fixture
    def processor(self, config):
        """Create a processor instance."""
        return AudioProcessor(config)

    def test_processor_start_stop(self, processor):
        """Test processor start/stop."""
        assert processor.start() is True
        assert processor._processing is True
        
        assert processor.stop() is True
        assert processor._processing is False

    def test_process_chunk(self, processor):
        """Test processing an audio chunk."""
        processor.start()
        
        # Set up mock storage callback
        storage_callback = Mock()
        processor.set_storage_callback(storage_callback)
        
        # Create test audio data
        chunk_data = np.random.randn(1600, 1).astype(np.float32)
        
        # Process event
        result = processor.process_chunk(1, chunk_data, 0.0, 0.1, -20.0)
        assert result is True
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify callback was called
        storage_callback.assert_called_once()
        
        processor.stop()

    def test_stats(self, processor):
        """Test processor statistics."""
        processor.start()
        
        chunk_data = np.random.randn(1600, 1).astype(np.float32)
        processor.process_chunk(1, chunk_data, 0.0, 0.1, -20.0)
        time.sleep(0.1)
        
        stats = processor.get_stats()
        assert stats["chunks_processed"] == 1
        assert stats["chunks_dropped"] == 0
        
        processor.stop()


class TestAudioStorage:
    """Tests for AudioStorage class."""

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
        return {
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
        }

    @pytest.fixture
    def storage(self, temp_dir, config):
        """Create storage instance."""
        return AudioStorage(temp_dir / "test-session", config)

    def test_storage_start_stop(self, storage):
        """Test storage start/stop."""
        assert storage.start() is True
        assert storage._writing is True
        
        assert storage.stop() is True
        assert storage._writing is False

    def test_write_chunk(self, storage):
        """Test writing an audio chunk."""
        storage.start()
        
        # Create test audio data (1 second of silence)
        chunk_data = np.zeros(16000, dtype=np.float32)
        
        result = storage.write_chunk(1, chunk_data, 0.0, 1.0, -60.0)
        assert result is True
        
        # Wait for write
        time.sleep(0.2)
        
        # Check files were created
        audio_dir = storage._audio_dir
        assert audio_dir.exists()
        
        # Check WAV file
        wav_files = list(audio_dir.glob("*.wav"))
        assert len(wav_files) == 1
        
        # Check index file
        index_file = audio_dir / "audio_index.jsonl"
        assert index_file.exists()
        
        # Read and verify index
        import orjson
        with open(index_file, 'rb') as f:
            line = f.readline()
            data = orjson.loads(line)
            assert data["chunk_index"] == 1
            assert data["sample_rate"] == 16000
        
        storage.stop()

    def test_multiple_chunks(self, storage):
        """Test writing multiple chunks."""
        storage.start()
        
        # Write multiple chunks
        for i in range(3):
            chunk_data = np.random.randn(16000, 1).astype(np.float32) * 0.1
            storage.write_chunk(i + 1, chunk_data, i * 1.0, 1.0, -20.0)
        
        time.sleep(0.3)
        
        # Verify all chunks written
        audio_dir = storage._audio_dir
        wav_files = list(audio_dir.glob("*.wav"))
        assert len(wav_files) == 3
        
        # Verify index
        import orjson
        index_file = audio_dir / "audio_index.jsonl"
        with open(index_file, 'rb') as f:
            lines = f.readlines()
            assert len(lines) == 3
        
        stats = storage.get_stats()
        assert stats["chunks_written"] == 3
        
        storage.stop()


class TestAudioCapture:
    """Tests for AudioCapture (modality implementation)."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "enabled": True,
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "chunk_duration_ms": 100,
            "device": None,
            "storage": {
                "output_directory": "data/sessions"
            }
        }

    @pytest.fixture
    def capture(self, config):
        """Create audio capture instance."""
        return AudioCapture(config)

    def test_capture_creation(self, capture):
        """Test capture creation."""
        assert capture.name == "audio"
        assert capture.status.value == "stopped"

    def test_capture_start_stop(self, capture, temp_dir):
        """Test capture start/stop."""
        config = capture.config.copy()
        config["storage"]["output_directory"] = str(temp_dir)
        
        # Mock device availability
        with patch('audio.capture.AudioListener.check_device_availability') as mock_check:
            mock_check.return_value = True
            
            # Mock sounddevice
            with patch('sounddevice.query_devices') as mock_devices:
                mock_devices.return_value = [
                    {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
                ]
                
                with patch('sounddevice.InputStream') as mock_stream_class:
                    mock_stream = MagicMock()
                    mock_stream_class.return_value = mock_stream
                    
                    # Mock unified event callback
                    unified_callback = Mock()
                    capture.set_unified_event_callback(unified_callback)
                    
                    # Start capture
                    result = capture.start("test-session", temp_dir)
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
        
        with patch('audio.capture.AudioListener.check_device_availability') as mock_check:
            mock_check.return_value = True
            
            with patch('sounddevice.query_devices') as mock_devices:
                mock_devices.return_value = [
                    {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
                ]
                
                with patch('sounddevice.InputStream') as mock_stream_class:
                    mock_stream = MagicMock()
                    mock_stream_class.return_value = mock_stream
                    
                    unified_callback = Mock()
                    capture.set_unified_event_callback(unified_callback)
                    
                    capture.start("test-session", temp_dir)
                    
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
        
        with patch('audio.capture.AudioListener.check_device_availability') as mock_check:
            mock_check.return_value = True
            
            with patch('sounddevice.query_devices') as mock_devices:
                mock_devices.return_value = [
                    {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
                ]
                
                with patch('sounddevice.InputStream') as mock_stream_class:
                    mock_stream = MagicMock()
                    mock_stream_class.return_value = mock_stream
                    
                    unified_callback = Mock()
                    capture.set_unified_event_callback(unified_callback)
                    
                    capture.start("test-session", temp_dir)
                    
                    stats = capture.get_stats()
                    assert "status" in stats
                    assert "processor" in stats
                    assert "storage" in stats
                    assert "listener" in stats
                    
                    capture.stop()


class TestAudioIntegration:
    """Integration tests for audio capture with session manager."""

    def test_audio_with_session_manager(self, temp_dir):
        """Test audio capture integrated with session manager.
        
        This tests the session lifecycle and modality registration.
        Note: WAV file creation is not tested here because we mock the audio input.
        """
        from core.config import get_default_config
        from session.manager import SessionManager
        
        config = get_default_config()
        config.storage.output_directory = str(temp_dir)
        config.audio.enabled = True
        config.keyboard.enabled = False
        config.camera.enabled = False
        
        with patch('audio.capture.AudioListener.check_device_availability') as mock_check:
            mock_check.return_value = True
            
            with patch('sounddevice.query_devices') as mock_devices:
                mock_devices.return_value = [
                    {'name': 'Test Device', 'max_input_channels': 1, 'default_samplerate': 48000}
                ]
                
                with patch('sounddevice.InputStream') as mock_stream_class:
                    mock_stream = MagicMock()
                    mock_stream_class.return_value = mock_stream
                    
                    manager = SessionManager(config=config, application_version="test-1.0.0")
                    
                    # Register audio capture
                    from audio import AudioCapture
                    audio_capture = AudioCapture(config.audio.__dict__)
                    audio_capture.set_unified_event_callback(
                        lambda event: manager._emit_event(event)
                    )
                    manager.register_modality(audio_capture)
                    
                    # Start session
                    result = manager.start_session()
                    assert result is True
                    assert manager.state.value == "recording"
                    
                    # Verify audio modality is running
                    assert "audio" in manager.modalities
                    assert manager.modalities["audio"].status.value == "running"
                    
                    # Wait a bit
                    time.sleep(0.2)
                    
                    # Stop session
                    result = manager.stop_session()
                    assert result is True
                    assert manager.state.value == "stopped"
                    
                    # Verify session directory was created
                    session_dir = temp_dir / manager.session_id
                    assert session_dir.exists()
                    
                    # Audio directory should be created by storage
                    audio_dir = session_dir / "audio"
                    assert audio_dir.exists()


# Need temp_dir fixture
@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)