"""
Interfaces and contracts for analysis modules.

Defines the common interfaces that all modality analysis modules must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BaselineStatus(str, Enum):
    """Status of baseline for a modality."""
    NOT_STARTED = "not_started"
    WARMING_UP = "warming_up"
    READY = "ready"
    ERROR = "error"


class DataQuality(str, Enum):
    """Data quality assessment."""
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    INSUFFICIENT = "insufficient"


@dataclass
class FeatureVector:
    """Generic feature vector with timestamp."""
    timestamp: float  # monotonic seconds since session start
    wallclock: str    # ISO 8601 UTC
    features: dict[str, float]
    modality: str
    quality: DataQuality = DataQuality.GOOD


@dataclass
class StateEstimate:
    """Estimated state with confidence and metadata."""
    state: str
    confidence: float  # 0.0 to 1.0
    timestamp: float
    wallclock: str
    contributing_features: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0  # 0.0 = certain, 1.0 = uncertain


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def extract(self, raw_data: Any, timestamp: float, wallclock: str) -> FeatureVector | None:
        """
        Extract features from raw modality data.

        Args:
            raw_data: Raw data from the modality (audio chunk, key events, face frame)
            timestamp: Monotonic timestamp (seconds since session start)
            wallclock: ISO 8601 UTC timestamp

        Returns:
            FeatureVector if extraction successful, None otherwise
        """

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""

    def reset(self) -> None:
        """Reset internal state (e.g., between sessions)."""


class StateEstimator(ABC):
    """Abstract base class for state estimators."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def estimate(self, features: FeatureVector, baseline_manager: "BaselineManager") -> StateEstimate | None:
        """
        Estimate state from features, normalized against baseline.

        Args:
            features: Feature vector from the modality
            baseline_manager: Baseline manager for this modality

        Returns:
            StateEstimate if estimation successful, None otherwise
        """

    @abstractmethod
    def get_possible_states(self) -> list[str]:
        """Return list of possible state labels."""

    def reset(self) -> None:
        """Reset internal state."""


class BaselineManager(ABC):
    """Abstract interface for baseline management."""

    @abstractmethod
    def update(self, feature_name: str, value: float, timestamp: float) -> None:
        """Update baseline with new observation."""

    @abstractmethod
    def get_baseline(self, feature_name: str) -> dict[str, float] | None:
        """Get current baseline statistics for a feature."""

    @abstractmethod
    def normalize(self, feature_name: str, value: float) -> float | None:
        """Normalize value against baseline (z-score). Returns None if baseline not ready."""

    @abstractmethod
    def is_ready(self, feature_name: str) -> bool:
        """Check if baseline is ready for a feature."""

    @abstractmethod
    def get_status(self, feature_name: str) -> BaselineStatus:
        """Get baseline status for a feature."""

    def reset(self) -> None:
        """Reset all baselines."""


class AnalysisResult(ABC):
    """Abstract interface for analysis results."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get monotonic timestamp."""

    @abstractmethod
    def get_modality(self) -> str:
        """Get modality name."""


class MultimodalFusion(ABC):
    """Abstract interface for multimodal fusion."""

    @abstractmethod
    def fuse(
        self,
        face_state: StateEstimate | None,
        audio_state: StateEstimate | None,
        keyboard_state: StateEstimate | None,
        baselines: dict[str, BaselineManager]
    ) -> StateEstimate | None:
        """
        Fuse modality-level states into a unified human state.

        Args:
            face_state: Face state estimate or None
            audio_state: Audio state estimate or None
            keyboard_state: Keyboard state estimate or None
            baselines: Dictionary of baseline managers per modality

        Returns:
            Fused StateEstimate or None if insufficient data
        """

    @abstractmethod
    def get_possible_states(self) -> list[str]:
        """Return list of possible fused state labels."""

    def reset(self) -> None:
        """Reset fusion state."""
