"""
Base feature extraction utilities.

Provides common utilities and base classes for modality-specific
feature extractors.
"""

import math
import time
from abc import abstractmethod
from datetime import UTC, datetime
from typing import Any

from analysis.interfaces import DataQuality, FeatureExtractor
from analysis.results import FeatureVector as FeatureVectorModel


class BaseFeatureExtractor(FeatureExtractor):
    """
    Base class for feature extractors with common functionality.
    
    Handles timestamp generation, quality assessment, and result formatting.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self._last_timestamp: float | None = None
        self._feature_names: list[str] = []

    def _create_feature_vector(
        self,
        features: dict[str, float],
        timestamp: float | None = None,
        wallclock: str | None = None,
        quality: DataQuality = DataQuality.GOOD,
        source_info: dict[str, Any] | None = None,
        processing_time_ms: float = 0.0
    ) -> FeatureVectorModel:
        """Create a standardized feature vector."""
        if timestamp is None:
            timestamp = time.monotonic()
        if wallclock is None:
            wallclock = datetime.now(UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        
        self._last_timestamp = timestamp
        
        return FeatureVectorModel(
            timestamp=timestamp,
            wallclock=wallclock,
            features=features,
            modality=self.name,
            quality=quality,
            source_info=source_info or {},
            processing_time_ms=processing_time_ms
        )

    def _assess_quality(self, features: dict[str, float]) -> DataQuality:
        """Assess data quality based on feature values."""
        if not features:
            return DataQuality.INSUFFICIENT
        
        # Check for NaN or infinite values
        for v in features.values():
            if math.isnan(v) or v == float('inf') or v == float('-inf'):
                return DataQuality.POOR
        
        return DataQuality.GOOD

    @abstractmethod
    def _extract_features(self, raw_data: Any) -> dict[str, float]:
        """Extract features from raw data. Must be implemented by subclasses."""

    def extract(
        self,
        raw_data: Any,
        timestamp: float | None = None,
        wallclock: str | None = None
    ) -> FeatureVectorModel | None:
        """Extract features with timing and quality assessment."""
        start_time = time.perf_counter()
        
        try:
            features = self._extract_features(raw_data)
            quality = self._assess_quality(features)
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return self._create_feature_vector(
                features=features,
                timestamp=timestamp,
                wallclock=wallclock,
                quality=quality,
                processing_time_ms=processing_time_ms
            )
        except (ValueError, TypeError, RuntimeError):
            # Return degraded quality feature vector on error
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            return self._create_feature_vector(
                features={},
                timestamp=timestamp,
                wallclock=wallclock,
                quality=DataQuality.POOR,
                processing_time_ms=processing_time_ms
            )


def compute_statistical_features(
    values: list[float],
    prefix: str = ""
) -> dict[str, float]:
    """
    Compute basic statistical features from a list of values.
    
    Args:
        values: List of numeric values
        prefix: Prefix for feature names
        
    Returns:
        Dict of statistical features
    """
    if not values:
        return {}
    
    n = len(values)
    mean = sum(values) / n
    
    features = {
        f"{prefix}count": float(n),
        f"{prefix}mean": mean,
        f"{prefix}min": min(values),
        f"{prefix}max": max(values),
        f"{prefix}range": max(values) - min(values),
    }
    
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        features[f"{prefix}std"] = variance ** 0.5
        features[f"{prefix}cv"] = features[f"{prefix}std"] / mean if mean != 0 else 0.0
    
    return features


def compute_rate_features(
    timestamps: list[float],
    prefix: str = ""
) -> dict[str, float]:
    """
    Compute rate features from timestamps.
    
    Args:
        timestamps: List of monotonic timestamps (seconds)
        prefix: Prefix for feature names
        
    Returns:
        Dict of rate features
    """
    if len(timestamps) < 2:
        return {}
    
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    
    if not intervals:
        return {}
    
    n = len(intervals)
    mean_interval = sum(intervals) / n
    
    features = {
        f"{prefix}rate_per_sec": n / (timestamps[-1] - timestamps[0]) if timestamps[-1] > timestamps[0] else 0.0,
        f"{prefix}mean_interval": mean_interval,
        f"{prefix}min_interval": min(intervals),
        f"{prefix}max_interval": max(intervals),
    }
    
    if n > 1:
        variance = sum((i - mean_interval) ** 2 for i in intervals) / (n - 1)
        features[f"{prefix}interval_std"] = variance ** 0.5
        features[f"{prefix}interval_cv"] = features[f"{prefix}interval_std"] / mean_interval if mean_interval != 0 else 0.0
    
    return features


def compute_trend(values: list[float], prefix: str = "") -> dict[str, float]:
    """
    Compute linear trend from values.
    
    Args:
        values: List of numeric values
        prefix: Prefix for feature names
        
    Returns:
        Dict with slope and r_squared
    """
    n = len(values)
    if n < 2:
        return {f"{prefix}slope": 0.0, f"{prefix}r_squared": 0.0}
    
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return {f"{prefix}slope": 0.0, f"{prefix}r_squared": 0.0}
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # R-squared
    ss_tot = sum((v - y_mean) ** 2 for v in values)
    ss_res = sum((values[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        f"{prefix}slope": slope,
        f"{prefix}r_squared": max(0.0, min(1.0, r_squared)),
    }