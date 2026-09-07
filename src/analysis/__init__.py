"""
Analysis Foundation Module

Provides reusable components for real-time human-state analysis:
- Temporal windows for rolling statistics
- Personal baseline normalization
- Feature extraction interfaces
- State estimation interfaces
- Analysis result schema
"""

from analysis.baseline import BaseBaselineManager
from analysis.interfaces import (
    AnalysisResult,
    BaselineManager,
    FeatureExtractor,
    StateEstimator,
)
from analysis.results import (
    AnalysisResult as AnalysisResultModel,
)
from analysis.results import (
    BaselineStatus,
    DataQuality,
    FeatureVector,
    StateEstimate,
)
from analysis.temporal import TemporalWindow

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
]