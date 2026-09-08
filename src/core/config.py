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
class AnalysisFaceConfig:
    enabled: bool = True
    window_size_sec: float = 2.0
    update_interval_ms: int = 200
    baseline_frames: int = 150
    expression_threshold: float = 0.3
    confidence_threshold: float = 0.5


@dataclass
class AnalysisAudioConfig:
    enabled: bool = True
    window_size_sec: float = 1.0
    hop_size_sec: float = 0.1
    baseline_duration_sec: float = 10.0
    pitch_min_hz: float = 80.0
    pitch_max_hz: float = 400.0
    pitch_threshold_semitones: float = 2.0
    energy_threshold_db: float = -40.0
    voice_activity_threshold: float = 0.01


@dataclass
class AnalysisKeyboardConfig:
    enabled: bool = True
    window_size_sec: float = 5.0
    baseline_events: int = 100
    correction_threshold_ms: int = 2000
    pause_threshold_sec: float = 2.0


@dataclass
class AnalysisFusionConfig:
    enabled: bool = True
    window_size_sec: float = 3.0
    update_interval_ms: int = 500
    min_modalities: int = 2
    confidence_threshold: float = 0.6
    uncertainty_threshold: float = 0.4
    modality_weights: dict[str, float] = field(default_factory=lambda: {
        "face": 1.0,
        "audio": 1.0,
        "keyboard": 1.0,
    })
    min_modalities_for_fusion: int = 2
    confidence_threshold: float = 0.6
    uncertainty_threshold: float = 0.4
    face_freshness_sec: float = 1.0
    audio_freshness_sec: float = 1.0
    keyboard_freshness_sec: float = 5.0
    face_stale_sec: float = 3.0
    audio_stale_sec: float = 3.0
    keyboard_stale_sec: float = 15.0
    smoothing_alpha: float = 0.3
    min_persistence_frames: int = 3
    debounce_sec: float = 2.0


@dataclass
class AnalysisStorageConfig:
    analysis_enabled: bool = True
    flush_interval_ms: int = 5000


@dataclass
class AnalysisConfig:
    enabled: bool = True
    temporal_window_sec: float = 3.0
    update_interval_ms: int = 500
    baseline_warmup_sec: float = 10.0
    baseline_min_samples: int = 50
    ema_alpha: float = 0.1
    min_std: float = 1e-6
    face: AnalysisFaceConfig = field(default_factory=AnalysisFaceConfig)
    audio: AnalysisAudioConfig = field(default_factory=AnalysisAudioConfig)
    keyboard: AnalysisKeyboardConfig = field(default_factory=AnalysisKeyboardConfig)
    fusion: AnalysisFusionConfig = field(default_factory=AnalysisFusionConfig)
    storage: AnalysisStorageConfig = field(default_factory=AnalysisStorageConfig)


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

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
            analysis=AnalysisConfig(
                **{k: v for k, v in data.get("analysis", {}).items() 
                   if k not in ["face", "audio", "keyboard", "fusion", "storage"]},
                face=AnalysisFaceConfig(**data.get("analysis", {}).get("face", {})),
                audio=AnalysisAudioConfig(**data.get("analysis", {}).get("audio", {})),
                keyboard=AnalysisKeyboardConfig(**data.get("analysis", {}).get("keyboard", {})),
                fusion=AnalysisFusionConfig(**data.get("analysis", {}).get("fusion", {})),
                storage=AnalysisStorageConfig(**data.get("analysis", {}).get("storage", {})),
            ),
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
            "analysis": {
                "enabled": self.analysis.enabled,
                "temporal_window_sec": self.analysis.temporal_window_sec,
                "update_interval_ms": self.analysis.update_interval_ms,
                "baseline_warmup_sec": self.analysis.baseline_warmup_sec,
                "baseline_min_samples": self.analysis.baseline_min_samples,
                "ema_alpha": self.analysis.ema_alpha,
                "min_std": self.analysis.min_std,
                "face": self.analysis.face.__dict__,
                "audio": self.analysis.audio.__dict__,
                "keyboard": self.analysis.keyboard.__dict__,
                "fusion": {
                    "enabled": self.analysis.fusion.enabled,
                    "window_size_sec": self.analysis.fusion.window_size_sec,
                    "update_interval_ms": self.analysis.fusion.update_interval_ms,
                    "min_modalities": self.analysis.fusion.min_modalities,
                    "confidence_threshold": self.analysis.fusion.confidence_threshold,
                    "uncertainty_threshold": self.analysis.fusion.uncertainty_threshold,
                    "modality_weights": self.analysis.fusion.modality_weights,
                    "min_modalities_for_fusion": self.analysis.fusion.min_modalities_for_fusion,
                    "confidence_threshold": self.analysis.fusion.confidence_threshold,
                    "uncertainty_threshold": self.analysis.fusion.uncertainty_threshold,
                    "face_freshness_sec": self.analysis.fusion.face_freshness_sec,
                    "audio_freshness_sec": self.analysis.fusion.audio_freshness_sec,
                    "keyboard_freshness_sec": self.analysis.fusion.keyboard_freshness_sec,
                    "face_stale_sec": self.analysis.fusion.face_stale_sec,
                    "audio_stale_sec": self.analysis.fusion.audio_stale_sec,
                    "keyboard_stale_sec": self.analysis.fusion.keyboard_stale_sec,
                    "smoothing_alpha": self.analysis.fusion.smoothing_alpha,
                    "min_persistence_frames": self.analysis.fusion.min_persistence_frames,
                    "debounce_sec": self.analysis.fusion.debounce_sec,
                },
                "storage": self.analysis.storage.__dict__,
            },
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