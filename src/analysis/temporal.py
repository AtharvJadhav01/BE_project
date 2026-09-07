"""
Temporal window for rolling statistics.

Provides a reusable component for maintaining rolling time windows
and computing basic statistics on timestamped observations.
"""

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class WindowStats:
    """Statistics computed from a temporal window."""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    latest: float = 0.0
    latest_timestamp: float = 0.0
    oldest_timestamp: float = 0.0

    def is_valid(self) -> bool:
        """Check if stats are valid (non-empty window)."""
        return self.count > 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean": self.mean if self.count > 0 else None,
            "std": self.std if self.count > 1 else None,
            "min": self.min_value if self.count > 0 else None,
            "max": self.max_value if self.count > 0 else None,
            "latest": self.latest if self.count > 0 else None,
            "latest_timestamp": self.latest_timestamp if self.count > 0 else None,
            "oldest_timestamp": self.oldest_timestamp if self.count > 0 else None,
        }


class TemporalWindow:
    """
    Rolling temporal window for streaming observations.

    Maintains observations within a configurable time window,
    automatically expiring old observations based on timestamps.
    Thread-safe for single-producer scenarios.
    """

    def __init__(
        self,
        window_duration_sec: float = 5.0,
        max_observations: int | None = None
    ):
        """
        Initialize temporal window.

        Args:
            window_duration_sec: Maximum time span of window in seconds
            max_observations: Optional hard limit on number of observations
        """
        if window_duration_sec <= 0:
            raise ValueError("window_duration_sec must be positive")
        
        self.window_duration_sec = window_duration_sec
        self.max_observations = max_observations
        
        # Store (timestamp, value) pairs
        self._observations: deque[tuple[float, float]] = deque()
        
        # Running statistics for O(1) updates
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def add(self, value: float, timestamp: float | None = None) -> None:
        """
        Add an observation to the window.

        Args:
            value: The numeric value to add
            timestamp: Monotonic timestamp (seconds). Uses time.monotonic() if None.
        """
        if timestamp is None:
            timestamp = time.monotonic()
        
        # Add new observation
        self._observations.append((timestamp, value))
        self._sum += value
        self._sum_sq += value * value
        self._count += 1
        
        # Enforce max observations limit
        if self.max_observations and len(self._observations) > self.max_observations:
            self._remove_oldest()
        
        # Expire old observations outside window
        self._expire_old(timestamp)

    def _remove_oldest(self) -> None:
        """Remove the oldest observation and update running stats."""
        if self._observations:
            _, old_val = self._observations.popleft()
            self._sum -= old_val
            self._sum_sq -= old_val * old_val
            self._count -= 1

    def _expire_old(self, current_time: float) -> None:
        """Remove observations older than window duration."""
        cutoff = current_time - self.window_duration_sec
        while self._observations and self._observations[0][0] < cutoff:
            self._remove_oldest()

    def get_stats(self) -> WindowStats:
        """
        Get current window statistics.

        Returns:
            WindowStats with count, mean, std, min, max, latest
        """
        if not self._observations:
            return WindowStats()
        
        stats = WindowStats()
        stats.count = self._count
        stats.mean = self._sum / self._count
        
        if self._count > 1:
            variance = (self._sum_sq / self._count) - (stats.mean * stats.mean)
            stats.std = math.sqrt(max(0.0, variance))
        
        stats.min_value = min(v for _, v in self._observations)
        stats.max_value = max(v for _, v in self._observations)
        stats.latest = self._observations[-1][1]
        stats.latest_timestamp = self._observations[-1][0]
        stats.oldest_timestamp = self._observations[0][0]
        
        return stats

    def get_values(self) -> list[float]:
        """Get all values in the window (oldest first)."""
        return [v for _, v in self._observations]

    def get_timestamps(self) -> list[float]:
        """Get all timestamps in the window (oldest first)."""
        return [t for t, _ in self._observations]

    def get_observations(self) -> list[tuple[float, float]]:
        """Get all (timestamp, value) pairs (oldest first)."""
        return list(self._observations)

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def __len__(self) -> int:
        """Return number of observations in window."""
        return len(self._observations)

    def is_empty(self) -> bool:
        """Check if window is empty."""
        return len(self._observations) == 0

    def get_span(self) -> float:
        """Get time span of current window (latest - oldest timestamp)."""
        if len(self._observations) < 2:
            return 0.0
        return self._observations[-1][0] - self._observations[0][0]


class MultiFeatureTemporalWindow:
    """
    Temporal window managing multiple features simultaneously.
    
    Useful for modalities that produce multiple feature values per observation.
    """

    def __init__(
        self,
        feature_names: list[str],
        window_duration_sec: float = 5.0,
        max_observations: int | None = None
    ):
        """
        Initialize multi-feature temporal window.

        Args:
            feature_names: List of feature names to track
            window_duration_sec: Maximum time span in seconds
            max_observations: Optional hard limit per feature
        """
        self.feature_names = feature_names
        self.windows = {
            name: TemporalWindow(window_duration_sec, max_observations)
            for name in feature_names
        }

    def add(self, features: dict[str, float], timestamp: float | None = None) -> None:
        """
        Add observations for multiple features.

        Args:
            features: Dict mapping feature names to values
            timestamp: Monotonic timestamp (shared across features)
        """
        for name, value in features.items():
            if name in self.windows:
                self.windows[name].add(value, timestamp)

    def get_stats(self) -> dict[str, WindowStats]:
        """Get statistics for all features."""
        return {name: window.get_stats() for name, window in self.windows.items()}

    def get_all_stats_dict(self) -> dict[str, dict]:
        """Get all statistics as dictionaries."""
        return {name: window.get_stats().to_dict() for name, window in self.windows.items()}

    def clear(self) -> None:
        """Clear all feature windows."""
        for window in self.windows.values():
            window.clear()

    def get_feature_names(self) -> list[str]:
        """Get list of tracked feature names."""
        return self.feature_names