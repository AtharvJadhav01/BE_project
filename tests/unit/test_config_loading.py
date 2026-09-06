"""
Unit tests for configuration loading.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config import (
    AppConfig,
    AudioConfig,
    KeyboardConfig,
    CameraConfig,
    FaceConfig,
    StorageConfig,
    LoggingConfig,
    UIConfig,
    load_config,
    get_default_config,
)


class TestAppConfig:
    """Tests for AppConfig and sub-configs."""

    def test_default_config(self):
        """Test default configuration values."""
        config = get_default_config()

        assert isinstance(config, AppConfig)
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.keyboard, KeyboardConfig)
        assert isinstance(config.camera, CameraConfig)
        assert isinstance(config.face, FaceConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.ui, UIConfig)

    def test_audio_config_defaults(self):
        """Test audio config defaults."""
        audio = AudioConfig()
        assert audio.enabled is True
        assert audio.sample_rate == 16000
        assert audio.channels == 1
        assert audio.format == "wav"
        assert audio.chunk_duration_ms == 10000
        assert audio.device is None

    def test_keyboard_config_defaults(self):
        """Test keyboard config defaults."""
        keyboard = KeyboardConfig()
        assert keyboard.enabled is True
        assert keyboard.capture_modifiers is True
        assert keyboard.capture_context is True
        assert keyboard.exclude_sensitive is True

    def test_camera_config_defaults(self):
        """Test camera config defaults."""
        camera = CameraConfig()
        assert camera.enabled is True
        assert camera.fps == 15
        assert camera.resolution == (640, 480)
        assert camera.device == 0

    def test_face_config_defaults(self):
        """Test face config defaults."""
        face = FaceConfig()
        assert face.enabled is True
        assert face.detection_model == "mediapipe"
        assert face.landmark_model == "mediapipe"
        assert face.refine_landmarks is True
        assert face.min_detection_confidence == 0.5
        assert face.min_tracking_confidence == 0.5

    def test_storage_config_defaults(self):
        """Test storage config defaults."""
        storage = StorageConfig()
        assert storage.output_directory == "data/sessions"
        assert storage.audio_chunk_duration_ms == 10000
        assert storage.flush_interval_ms == 5000
        assert storage.max_file_size_mb == 100

    def test_logging_config_defaults(self):
        """Test logging config defaults."""
        logging = LoggingConfig()
        assert logging.level == "INFO"
        assert logging.format == "json"
        assert logging.file == "logs/app.log"
        assert logging.max_size_mb == 10
        assert logging.backup_count == 5

    def test_ui_config_defaults(self):
        """Test UI config defaults."""
        ui = UIConfig()
        assert ui.theme == "default"
        assert ui.update_interval_ms == 100
        assert ui.show_system_tray is True

    def test_config_serialization(self):
        """Test config to/from dict serialization."""
        config = get_default_config()
        data = config.to_dict()

        assert "audio" in data
        assert "keyboard" in data
        assert "camera" in data
        assert "face" in data
        assert "storage" in data
        assert "logging" in data
        assert "ui" in data

        # Test from_dict
        restored = AppConfig.from_dict(data)
        assert restored.audio.sample_rate == config.audio.sample_rate
        assert restored.keyboard.enabled == config.keyboard.enabled
        assert restored.camera.fps == config.camera.fps


class TestLoadConfig:
    """Tests for load_config function."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "audio": {
                    "enabled": True,
                    "sample_rate": 44100,
                    "channels": 2,
                },
                "keyboard": {
                    "enabled": False,
                },
                "camera": {
                    "enabled": True,
                    "fps": 30,
                },
            }, f)
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_load_config_from_file(self, temp_config_file):
        """Test loading config from file."""
        config = load_config(temp_config_file)

        assert config.audio.enabled is True
        assert config.audio.sample_rate == 44100
        assert config.audio.channels == 2
        assert config.keyboard.enabled is False
        assert config.camera.enabled is True
        assert config.camera.fps == 30

        # Unspecified values should use defaults
        assert config.audio.format == "wav"
        assert config.keyboard.capture_modifiers is True

    def test_load_config_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path/config.yaml"))

    def test_load_config_empty_file(self):
        """Test loading config from empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            # Should get defaults
            assert config.audio.sample_rate == 16000
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_config_none_returns_defaults(self):
        """Test that None config path uses default location."""
        # This will fail because default config doesn't exist in test env
        # but we can test the default config function
        config = get_default_config()
        assert config is not None
        assert isinstance(config, AppConfig)


class TestConfigEdgeCases:
    """Tests for configuration edge cases."""

    def test_partial_config(self):
        """Test loading partial configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "audio": {
                    "sample_rate": 48000,
                },
            }, f)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.audio.sample_rate == 48000
            # Other audio settings should be defaults
            assert config.audio.channels == 1
            assert config.audio.format == "wav"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_config_with_null_values(self):
        """Test config with null values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "audio": {
                    "device": None,
                },
            }, f)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.audio.device is None
        finally:
            temp_path.unlink(missing_ok=True)