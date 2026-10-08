"""
Analysis Foundation Module

Provides reusable components for real-time human-state analysis:
- Temporal windows for rolling statistics
- Personal baseline normalization
- Feature extraction interfaces
- State estimation interfaces
- Analysis result schema
- Multimodal fusion for human-state estimation
"""

from .baseline import BaseBaselineManager
from .interfaces import (
    AnalysisResult,
    BaselineManager,
    FeatureExtractor,
    StateEstimator,
)
from .results import (
    AnalysisResult as AnalysisResultModel,
)
from .results import (
    BaselineStatus,
    DataQuality,
    FeatureVector,
    StateEstimate,
)
from .temporal import TemporalWindow
from .multimodal import (
    MultimodalFusion,
    HumanStateEstimator,
    TemporalAligner,
    ModalitySnapshot,
    FusionEngine,
)

__all__ = [
    "AnalysisResult",
    "AnalysisResultModel",
    "BaseBaselineManager",
    "BaselineManager",
    "BaselineStatus",
    "DataQuality",
    "FeatureExtractor",
    "FeatureVector",
    "StateEstimate",
    "StateEstimator",
    "TemporalWindow",
    "MultimodalFusion",
    "HumanStateEstimator",
    "TemporalAligner",
    "ModalitySnapshot",
    "FusionEngine",
]