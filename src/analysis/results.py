"""
Analysis result data models.

Provides structured data classes for analysis output that can be serialized
and stored. Extends the interfaces with concrete implementations.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from analysis.interfaces import (
    BaselineStatus,
    DataQuality,
)
from analysis.interfaces import (
    FeatureVector as BaseFeatureVector,
)
from analysis.interfaces import (
    StateEstimate as BaseStateEstimate,
)


@dataclass
class FeatureVector(BaseFeatureVector):
    """Concrete feature vector with additional metadata."""
    # Core fields from base
    timestamp: float = 0.0
    wallclock: str = ""
    features: dict[str, float] = field(default_factory=dict)
    modality: str = ""
    quality: DataQuality = DataQuality.GOOD
    
    # Additional metadata
    source_info: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "wallclock": self.wallclock,
            "features": self.features,
            "modality": self.modality,
            "quality": self.quality.value,
            "source_info": self.source_info,
            "processing_time_ms": self.processing_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureVector":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            wallclock=data["wallclock"],
            features=data["features"],
            modality=data["modality"],
            quality=DataQuality(data.get("quality", "good")),
            source_info=data.get("source_info", {}),
            processing_time_ms=data.get("processing_time_ms", 0.0),
        )


@dataclass
class StateEstimate(BaseStateEstimate):
    """Concrete state estimate with additional metadata."""
    # Core fields from base
    state: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0
    wallclock: str = ""
    contributing_features: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0
    
    # Additional metadata
    modality: str = ""
    baseline_status: BaselineStatus = BaselineStatus.NOT_STARTED
    raw_features: dict[str, float] = field(default_factory=dict)
    normalized_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "wallclock": self.wallclock,
            "contributing_features": self.contributing_features,
            "uncertainty": self.uncertainty,
            "modality": self.modality,
            "baseline_status": self.baseline_status.value,
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateEstimate":
        """Create from dictionary."""
        return cls(
            state=data["state"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            wallclock=data["wallclock"],
            contributing_features=data.get("contributing_features", {}),
            uncertainty=data.get("uncertainty", 0.0),
            modality=data.get("modality", ""),
            baseline_status=BaselineStatus(data.get("baseline_status", "not_started")),
            raw_features=data.get("raw_features", {}),
            normalized_features=data.get("normalized_features", {}),
        )


@dataclass
class AnalysisResult:
    """Complete analysis result for a modality at a timestamp."""
    modality: str
    timestamp: float
    wallclock: str
    feature_vector: FeatureVector
    state_estimate: StateEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "modality": self.modality,
            "timestamp": self.timestamp,
            "wallclock": self.wallclock,
            "feature_vector": self.feature_vector.to_dict(),
            "metadata": self.metadata,
        }
        if self.state_estimate:
            result["state_estimate"] = self.state_estimate.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        """Create from dictionary."""
        fv = FeatureVector.from_dict(data["feature_vector"])
        se = None
        if data.get("state_estimate"):
            se = StateEstimate.from_dict(data["state_estimate"])
        return cls(
            modality=data["modality"],
            timestamp=data["timestamp"],
            wallclock=data["wallclock"],
            feature_vector=fv,
            state_estimate=se,
            metadata=data.get("metadata", {}),
        )

    def get_timestamp(self) -> float:
        """Get monotonic timestamp."""
        return self.timestamp

    def get_modality(self) -> str:
        """Get modality name."""
        return self.modality


@dataclass
class MultimodalAnalysisResult:
    """Fused analysis result from multiple modalities."""
    timestamp: float
    wallclock: str
    modality_results: dict[str, AnalysisResult] = field(default_factory=dict)
    fused_state: StateEstimate | None = None
    fusion_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "wallclock": self.wallclock,
            "modality_results": {
                name: result.to_dict() for name, result in self.modality_results.items()
            },
            "fused_state": self.fused_state.to_dict() if self.fused_state else None,
            "fusion_metadata": self.fusion_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultimodalAnalysisResult":
        """Create from dictionary."""
        modality_results = {}
        for name, result_data in data.get("modality_results", {}).items():
            modality_results[name] = AnalysisResult.from_dict(result_data)
        
        fused_state = None
        if data.get("fused_state"):
            fused_state = StateEstimate.from_dict(data["fused_state"])
        
        return cls(
            timestamp=data["timestamp"],
            wallclock=data["wallclock"],
            modality_results=modality_results,
            fused_state=fused_state,
            fusion_metadata=data.get("fusion_metadata", {}),
        )


@dataclass
class AnalysisConfig:
    """Configuration for analysis modules."""
    temporal_window_sec: float = 3.0
    update_interval_ms: int = 500
    baseline_warmup_sec: float = 10.0
    baseline_min_samples: int = 50
    ema_alpha: float = 0.1
    confidence_threshold: float = 0.6
    uncertainty_threshold: float = 0.4
    min_modalities_for_fusion: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})