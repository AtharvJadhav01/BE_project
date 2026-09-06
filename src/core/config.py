"""
Configuration loading and management for the application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    chunk_duration_ms: int = 10000
    device: Optional[int] = None


@dataclass
class KeyboardConfig:
    enabled: bool = True
    capture_modifiers: bool = True
    capture_context: bool = True
    exclude_sensitive: bool = True


@dataclass
class CameraConfig:
    enabled: bool = True
    fps: int = 15
    resolution: tuple[int, int] = (640, 480)
    device: int = 0


@dataclass
class FaceConfig:
    enabled: bool = True
    detection_model: str = "mediapipe"
    landmark_model: str = "mediapipe"
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class StorageConfig:
    output_directory: str = "data/sessions"
    audio_chunk_duration_ms: int = 10000
    flush_interval_ms: int = 5000
    max_file_size_mb: int = 100


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/app.log"
    max_size_mb: int = 10
    backup_count: int = 5


@dataclass
class UIConfig:
    theme: str = "default"
    update_interval_ms: int = 100
    show_system_tray: bool = True


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Create AppConfig from dictionary."""
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            keyboard=KeyboardConfig(**data.get("keyboard", {})),
            camera=CameraConfig(**data.get("camera", {})),
            face=FaceConfig(**data.get("face", {})),
            storage=StorageConfig(**data.get("storage", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            ui=UIConfig(**data.get("ui", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            "audio": self.audio.__dict__,
            "keyboard": self.keyboard.__dict__,
            "camera": self.camera.__dict__,
            "face": self.face.__dict__,
            "storage": self.storage.__dict__,
            "logging": self.logging.__dict__,
            "ui": self.ui.__dict__,
        }


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config/config.yaml

    Returns:
        AppConfig object with loaded configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return AppConfig.from_dict(data)


def get_default_config() -> AppConfig:
    """Get default configuration without loading from file."""
    return AppConfig()