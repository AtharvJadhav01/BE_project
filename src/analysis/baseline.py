"""
Personal baseline management for modality features.

Provides reusable baseline tracking with warm-up period,
exponential moving average adaptation, and z-score normalization.
"""

import math
import time
from dataclasses import dataclass

from analysis.interfaces import BaselineManager, BaselineStatus


@dataclass
class FeatureBaseline:
    """Baseline statistics for a single feature."""
    name: str
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0  # Population variance
    ema: float = 0.0       # Exponential moving average
    last_update: float = 0.0
    status: BaselineStatus = BaselineStatus.NOT_STARTED

    @property
    def std(self) -> float:
        """Standard deviation (sample standard deviation for z-score normalization)."""
        if self.count < 2:
            return 0.0
        # Welford's algorithm accumulates M2 (sum of squared differences)
        # Sample variance = M2 / (n - 1)
        sample_variance = self.variance / (self.count - 1)
        return math.sqrt(max(0.0, sample_variance))

    @property
    def is_ready(self) -> bool:
        """Check if baseline has sufficient samples."""
        return bool(self.status == BaselineStatus.READY)

    def update_welford(self, value: float) -> None:
        """
        Update mean and variance using Welford's online algorithm.
        
        Numerically stable for streaming updates.
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.variance += delta * delta2

    def update_ema(self, value: float, alpha: float) -> None:
        """
        Update exponential moving average.
        
        Args:
            value: New observation
            alpha: EMA decay factor (0 < alpha <= 1). Higher = more responsive.
        """
        if self.count == 1:
            self.ema = value
        else:
            self.ema = alpha * value + (1 - alpha) * self.ema


class BaseBaselineManager(BaselineManager):
    """
    Personal baseline manager with warm-up and EMA adaptation.

    Features:
    - Warm-up period to collect initial baseline
    - Welford's algorithm for numerically stable mean/variance
    - Exponential moving average for adaptive baseline
    - Z-score normalization against personal baseline
    - Per-feature baseline tracking
    - Thread-safe for single-producer scenarios
    """

    def __init__(
        self,
        warmup_duration_sec: float = 10.0,
        min_samples: int = 50,
        ema_alpha: float = 0.1,
        min_std: float = 1e-6
    ):
        """
        Initialize baseline manager.

        Args:
            warmup_duration_sec: Duration to collect baseline before ready
            min_samples: Minimum samples required before baseline is ready
            ema_alpha: EMA decay factor (0 < alpha <= 1). Higher = faster adaptation.
            min_std: Minimum standard deviation to avoid division by zero
        """
        if warmup_duration_sec <= 0:
            raise ValueError("warmup_duration_sec must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if min_std <= 0:
            raise ValueError("min_std must be positive")

        self.warmup_duration_sec = warmup_duration_sec
        self.min_samples = min_samples
        self.ema_alpha = ema_alpha
        self.min_std = min_std

        self._baselines: dict[str, FeatureBaseline] = {}
        self._start_time: float | None = None

    def _get_or_create(self, feature_name: str) -> FeatureBaseline:
        """Get or create baseline for a feature."""
        if feature_name not in self._baselines:
            self._baselines[feature_name] = FeatureBaseline(name=feature_name)
        return self._baselines[feature_name]

    def _ensure_started(self, timestamp: float) -> None:
        """Record start time on first update."""
        if self._start_time is None:
            self._start_time = timestamp

    def update(self, feature_name: str, value: float, timestamp: float | None = None) -> None:
        """
        Update baseline with new observation.

        Args:
            feature_name: Name of the feature
            value: Observed value
            timestamp: Monotonic timestamp. Uses time.monotonic() if None.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        self._ensure_started(timestamp)
        baseline = self._get_or_create(feature_name)

        # Update using Welford's algorithm (numerically stable)
        baseline.update_welford(value)
        baseline.update_ema(value, self.ema_alpha)
        baseline.last_update = timestamp
        baseline.count += 0  # Already incremented in update_welford

        # Update status
        elapsed = timestamp - (self._start_time or 0.0)
        if baseline.status == BaselineStatus.NOT_STARTED:
            baseline.status = BaselineStatus.WARMING_UP
        
        if baseline.count >= self.min_samples and elapsed >= self.warmup_duration_sec:
            baseline.status = BaselineStatus.READY

    def get_baseline(self, feature_name: str) -> dict[str, float] | None:
        """
        Get current baseline statistics for a feature.

        Returns:
            Dict with mean, std, count, ema, status or None if not tracked
        """
        baseline = self._baselines.get(feature_name)
        if not baseline:
            return None

        return {
            "mean": baseline.mean,
            "std": baseline.std,
            "count": baseline.count,
            "ema": baseline.ema,
            "status": baseline.status.value,
            "min_std": self.min_std,
        }

    def normalize(self, feature_name: str, value: float) -> float | None:
        """
        Normalize value against baseline (z-score).

        Args:
            feature_name: Name of the feature
            value: Value to normalize

        Returns:
            Z-score (value - mean) / std, or None if baseline not ready
        """
        baseline = self._baselines.get(feature_name)
        if not baseline or not baseline.is_ready:
            return None

        std = max(baseline.std, self.min_std)
        return (value - baseline.mean) / std

    def get_ema(self, feature_name: str) -> float | None:
        """Get exponential moving average for a feature."""
        baseline = self._baselines.get(feature_name)
        return baseline.ema if baseline else None

    def is_ready(self, feature_name: str) -> bool:
        """Check if baseline is ready for a feature."""
        baseline = self._baselines.get(feature_name)
        return baseline.is_ready if baseline else False

    def get_status(self, feature_name: str) -> BaselineStatus:
        """Get baseline status for a feature."""
        baseline = self._baselines.get(feature_name)
        return baseline.status if baseline else BaselineStatus.NOT_STARTED

    def get_all_statuses(self) -> dict[str, BaselineStatus]:
        """Get status for all tracked features."""
        return {name: bl.status for name, bl in self._baselines.items()}

    def get_all_baselines(self) -> dict[str, dict[str, float]]:
        """Get baseline stats for all features."""
        result = {}
        for name in self._baselines:
            baseline_dict = self.get_baseline(name)
            if baseline_dict is not None:
                result[name] = baseline_dict
        return result

    def get_warmup_progress(self, timestamp: float | None = None) -> float:
        """
        Get warm-up progress as 0.0 to 1.0.

        Returns:
            Progress toward baseline readiness (based on time and samples)
        """
        if timestamp is None:
            timestamp = time.monotonic()
        
        if self._start_time is None:
            return 0.0

        elapsed = timestamp - self._start_time
        time_progress = min(1.0, elapsed / self.warmup_duration_sec)
        
        # Overall progress is min of time and sample progress
        max_samples = max(b.count for b in self._baselines.values()) if self._baselines else 0
        sample_progress = min(1.0, max_samples / self.min_samples) if self.min_samples > 0 else 1.0
        
        return min(time_progress, sample_progress)

    def reset(self) -> None:
        """Reset all baselines."""
        self._baselines.clear()
        self._start_time = None

    def reset_feature(self, feature_name: str) -> None:
        """Reset baseline for a specific feature."""
        if feature_name in self._baselines:
            del self._baselines[feature_name]