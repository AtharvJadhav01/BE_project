"""
Keyboard feature extraction for real-time typing behavior analysis.

Converts raw keyboard events into meaningful behavioral features including
typing speed, inter-key intervals, corrections, pauses, and rhythm metrics.
"""

import math
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

from analysis.interfaces import FeatureExtractor, FeatureVector, DataQuality
from analysis.results import FeatureVector as FeatureVectorModel
from analysis.features import BaseFeatureExtractor, compute_statistical_features, compute_rate_features, compute_trend


@dataclass
class KeyboardFeatures:
    """Container for extracted keyboard features."""
    # Typing speed features
    typing_speed_kpm: float = 0.0  # keys per minute
    typing_speed_kps: float = 0.0  # keys per second
    
    # Inter-key interval features
    iki_mean: float = 0.0      # mean inter-key interval (ms)
    iki_std: float = 0.0       # standard deviation of IKI (ms)
    iki_min: float = 0.0       # minimum IKI (ms)
    iki_max: float = 0.0       # maximum IKI (ms)
    iki_cv: float = 0.0        # coefficient of variation of IKI
    
    # Key duration features
    key_duration_mean: float = 0.0   # mean key hold duration (ms)
    key_duration_std: float = 0.0    # std of key hold duration (ms)
    key_duration_min: float = 0.0    # minimum key hold (ms)
    key_duration_max: float = 0.0    # maximum key hold (ms)
    
    # Correction features
    backspace_count: int = 0
    delete_count: int = 0
    correction_count: int = 0           # total corrections (backspace + delete)
    correction_rate: float = 0.0       # corrections per minute
    correction_burst_count: int = 0    # consecutive corrections
    
    # Pause features
    pause_count: int = 0
    pause_duration_mean: float = 0.0   # mean pause duration (ms)
    pause_duration_max: float = 0.0    # maximum pause (ms)
    long_pause_count: int = 0          # pauses > threshold
    
    # Rhythm/regularity features
    typing_rhythm_regularity: float = 0.0  # 0-1, 1 = perfectly regular
    burst_typing_score: float = 0.0        # 0-1, 1 = very bursty
    hesitation_score: float = 0.0          # 0-1, 1 = highly hesitant
    
    # Overall activity
    total_keys: int = 0
    printable_keys: int = 0
    modifier_usage: float = 0.0            # fraction of keys with modifiers
    
    # Activity state
    is_active: bool = False
    time_since_last_key: float = 0.0       # seconds since last key


class KeyboardFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts behavioral features from keyboard event stream.
    
    Converts raw keyboard events (key_down, key_up) into meaningful
    behavioral features that capture typing patterns without
    exposing sensitive content.
    """
    
    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration
        self.typing_speed_window_sec = config.get("typing_speed_window_sec", 30.0)
        self.correction_window_sec = config.get("correction_window_sec", 5.0)
        self.pause_threshold_ms = config.get("pause_threshold_ms", 2000)  # 2 seconds
        self.long_pause_threshold_ms = config.get("long_pause_threshold_ms", 5000)  # 5 seconds
        self.correction_burst_threshold_ms = config.get("correction_burst_threshold_ms", 2000)
        
        # State for feature computation
        self._key_timestamps: List[float] = []          # timestamps of key presses
        self._key_durations: List[float] = []           # key hold durations (ms)
        self._inter_key_intervals: List[float] = []     # intervals between keys (ms)
        self._pause_durations: List[float] = []         # pause durations (ms)
        self._key_names: List[str] = []                 # key names for correction detection
        self._last_key_time: Optional[float] = None
        self._last_key_name: Optional[str] = None
        self._pressed_keys: Dict[str, float] = {}       # key -> press time
        self._total_keys: int = 0
        self._printable_keys: int = 0
        self._modifier_keys: int = 0
        self._correction_keys: set = {"Backspace", "Delete"}
        
        # Correction tracking
        self._recent_corrections: List[float] = []      # timestamps of corrections
        self._correction_burst_count: int = 0
        self._in_correction_burst: bool = False
        
        # Pause tracking
        self._in_pause: bool = False
        self._pause_start_time: Optional[float] = None
        
    def _extract_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract keyboard features from raw keyboard event data.
        
        Args:
            raw_data: Dictionary containing keyboard event data with:
                     - event_type: "key_down" or "key_up"
                     - key: key name
                     - is_printable: bool
                     - modifiers: list of active modifiers
                     - duration_ms: key hold duration (for key_up)
                     - is_sensitive_context: bool
        
        Returns:
            Dictionary of keyboard behavioral features
        """
        event_type = raw_data.get("event_type", "")
        key_name = raw_data.get("key", "")
        is_printable = raw_data.get("is_printable", False)
        modifiers = raw_data.get("modifiers", [])
        duration_ms = raw_data.get("duration_ms")
        is_sensitive = raw_data.get("is_sensitive_context", False)
        
        current_time = time.monotonic()
        features = KeyboardFeatures()
        
        # Update internal state based on event
        if event_type == "key_down":
            self._process_key_down(key_name, is_printable, modifiers, current_time)
        elif event_type == "key_up":
            self._process_key_up(key_name, duration_ms, current_time)
        
        # Compute current features from accumulated state
        self._compute_features(features, current_time)
        
        # Convert to dictionary
        return self._features_to_dict(features)
    
    def _process_key_down(self, key_name: str, is_printable: bool, modifiers: List[str], current_time: float) -> None:
        """Process key down event."""
        self._total_keys += 1
        self._key_timestamps.append(current_time)
        
        if key_name in self._correction_keys:
            # Track correction key
            self._recent_corrections.append(current_time)
            # Clean old corrections outside window
            cutoff = current_time - self.correction_window_sec
            self._recent_corrections = [t for t in self._recent_corrections if t > cutoff]
        
        if is_printable:
            self._printable_keys += 1
        
        if modifiers:
            self._modifier_keys += 1
        
        # Track inter-key interval
        if self._last_key_time is not None:
            iki = (current_time - self._last_key_time) * 1000  # ms
            self._inter_key_intervals.append(iki)
            
            # Check for pause
            if iki > self.pause_threshold_ms:
                self._pause_durations.append(iki)
                if iki > self.long_pause_threshold_ms:
                    pass  # counted in compute_features
        
        # Track key press for duration calculation
        self._pressed_keys[key_name] = current_time
        
        self._last_key_time = current_time
        self._last_key_name = key_name
        self._key_names.append(key_name)
        
        # Track correction bursts
        if key_name in self._correction_keys:
            self._track_correction_burst(current_time)
    
    def _process_key_up(self, key_name: str, duration_ms: Optional[float], current_time: float) -> None:
        """Process key up event."""
        if duration_ms is not None and duration_ms > 0:
            self._key_durations.append(duration_ms)
        
        # Remove from pressed keys
        self._pressed_keys.pop(key_name, None)
    
    def _track_correction_burst(self, current_time: float) -> None:
        """Track consecutive corrections."""
        if not self._in_correction_burst:
            # Check if previous correction was recent
            if self._recent_corrections and len(self._recent_corrections) >= 2:
                last_two = self._recent_corrections[-2:]
                if (last_two[1] - last_two[0]) * 1000 < self.correction_burst_threshold_ms:
                    self._in_correction_burst = True
                    self._correction_burst_count = 2
        else:
            # Continue tracking burst
            if self._recent_corrections and len(self._recent_corrections) >= 1:
                last_correction = self._recent_corrections[-1]
                # Check if this is part of the same burst
                if self._recent_corrections[-2:] and len(self._recent_corrections) >= 2:
                    last_two = self._recent_corrections[-2:]
                    if (last_two[1] - last_two[0]) * 1000 < self.correction_burst_threshold_ms:
                        self._correction_burst_count += 1
                    else:
                        self._in_correction_burst = False
                        self._correction_burst_count = 0
                else:
                    self._in_correction_burst = False
                    self._correction_burst_count = 0
    
    def _compute_features(self, features: KeyboardFeatures, current_time: float) -> None:
        """Compute features from accumulated state."""
        features.total_keys = self._total_keys
        features.printable_keys = self._printable_keys
        features.modifier_usage = self._modifier_keys / max(1, self._total_keys)
        
        # Time since last key
        if self._last_key_time:
            features.time_since_last_key = current_time - self._last_key_time
            features.is_active = features.time_since_last_key < 5.0  # Active if key in last 5s
        else:
            features.time_since_last_key = 0.0
            features.is_active = False
        
        # Typing speed (keys per minute/second)
        if len(self._key_timestamps) > 1:
            time_span = self._key_timestamps[-1] - self._key_timestamps[0]
            if time_span > 0:
                features.typing_speed_kps = len(self._key_timestamps) / time_span
                features.typing_speed_kpm = features.typing_speed_kps * 60
        
        # Inter-key interval statistics
        if self._inter_key_intervals:
            iki_stats = compute_statistical_features(self._inter_key_intervals, "iki_")
            features.iki_mean = iki_stats.get("iki_mean", 0.0)
            features.iki_std = iki_stats.get("iki_std", 0.0)
            features.iki_min = iki_stats.get("iki_min", 0.0)
            features.iki_max = iki_stats.get("iki_max", 0.0)
            features.iki_cv = iki_stats.get("iki_cv", 0.0)
        
        # Key duration statistics
        if self._key_durations:
            dur_stats = compute_statistical_features(self._key_durations, "key_duration_")
            features.key_duration_mean = dur_stats.get("key_duration_mean", 0.0)
            features.key_duration_std = dur_stats.get("key_duration_std", 0.0)
            features.key_duration_min = dur_stats.get("key_duration_min", 0.0)
            features.key_duration_max = dur_stats.get("key_duration_max", 0.0)
        
        # Correction statistics
        features.backspace_count = sum(1 for k in self._key_names if k == "Backspace")
        features.delete_count = sum(1 for k in self._key_names if k == "Delete")
        features.correction_count = features.backspace_count + features.delete_count
        
        if self._key_timestamps and len(self._key_timestamps) > 1:
            time_span_min = (self._key_timestamps[-1] - self._key_timestamps[0]) / 60.0
            if time_span_min > 0:
                features.correction_rate = features.correction_count / time_span_min
        
        features.correction_burst_count = self._correction_burst_count
        
        # Pause statistics
        features.pause_count = len(self._pause_durations)
        if self._pause_durations:
            pause_stats = compute_statistical_features(self._pause_durations, "pause_")
            features.pause_duration_mean = pause_stats.get("pause_mean", 0.0)
            features.pause_duration_max = pause_stats.get("pause_max", 0.0)
        
        features.long_pause_count = sum(1 for p in self._pause_durations if p > self.long_pause_threshold_ms)
        
        # Rhythm/regularity features
        if len(self._inter_key_intervals) > 2:
            # Regularity: inverse of coefficient of variation
            cv = features.iki_cv
            features.typing_rhythm_regularity = max(0.0, 1.0 - min(1.0, cv))
            
            # Burst typing: detect clusters of rapid typing
            features.burst_typing_score = self._compute_burst_score()
            
            # Hesitation: long IKIs and pauses
            features.hesitation_score = self._compute_hesitation_score()
        
        # Set activity state
        features.is_active = self._last_key_time is not None and (time.monotonic() - self._last_key_time) < 5.0
        if self._last_key_time:
            features.time_since_last_key = time.monotonic() - self._last_key_time
    
    def _compute_burst_score(self) -> float:
        """Compute burst typing score (0-1, higher = more bursty)."""
        if len(self._inter_key_intervals) < 3:
            return 0.0
        
        # Detect clusters of short intervals followed by long pauses
        burst_count = 0
        in_burst = False
        for iki in self._inter_key_intervals:
            if iki < 150:  # Fast typing (< 150ms between keys)
                if not in_burst:
                    in_burst = True
                    burst_count += 1
            else:
                in_burst = False
        
        # Normalize by total intervals
        return min(1.0, burst_count / max(1, len(self._inter_key_intervals) / 5))
    
    def _compute_hesitation_score(self) -> float:
        """Compute hesitation score (0-1, higher = more hesitant)."""
        if len(self._inter_key_intervals) < 2:
            return 0.0
        
        # Hesitation = long intervals + long pauses relative to total
        long_intervals = sum(1 for iki in self._inter_key_intervals if iki > 500)
        long_pauses = sum(1 for p in self._pause_durations if p > 2000)
        total_events = len(self._inter_key_intervals) + len(self._pause_durations)
        
        if total_events == 0:
            return 0.0
        
        hesitation = (long_intervals + long_pauses) / total_events
        return min(1.0, hesitation * 2)  # Scale up
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no events available."""
        features = KeyboardFeatures()
        return self._features_to_dict(features)
    
    def _features_to_dict(self, features: KeyboardFeatures) -> Dict[str, float]:
        """Convert KeyboardFeatures dataclass to dictionary."""
        return {
            # Typing speed
            "typing_speed_kpm": features.typing_speed_kpm,
            "typing_speed_kps": features.typing_speed_kps,
            
            # Inter-key interval
            "iki_mean": features.iki_mean,
            "iki_std": features.iki_std,
            "iki_min": features.iki_min,
            "iki_max": features.iki_max,
            "iki_cv": features.iki_cv,
            
            # Key duration
            "key_duration_mean": features.key_duration_mean,
            "key_duration_std": features.key_duration_std,
            "key_duration_min": features.key_duration_min,
            "key_duration_max": features.key_duration_max,
            
            # Corrections
            "backspace_count": float(features.backspace_count),
            "delete_count": float(features.delete_count),
            "correction_count": float(features.correction_count),
            "correction_rate": features.correction_rate,
            "correction_burst_count": float(features.correction_burst_count),
            
            # Pauses
            "pause_count": float(features.pause_count),
            "pause_duration_mean": features.pause_duration_mean,
            "pause_duration_max": features.pause_duration_max,
            "long_pause_count": float(features.long_pause_count),
            
            # Rhythm
            "typing_rhythm_regularity": features.typing_rhythm_regularity,
            "burst_typing_score": features.burst_typing_score,
            "hesitation_score": features.hesitation_score,
            
            # Overall
            "total_keys": float(features.total_keys),
            "printable_keys": float(features.printable_keys),
            "modifier_usage": features.modifier_usage,
            "is_active": 1.0 if features.is_active else 0.0,
            "time_since_last_key": features.time_since_last_key,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            # Typing speed
            "typing_speed_kpm", "typing_speed_kps",
            
            # Inter-key interval
            "iki_mean", "iki_std", "iki_min", "iki_max", "iki_cv",
            
            # Key duration
            "key_duration_mean", "key_duration_std", "key_duration_min", "key_duration_max",
            
            # Corrections
            "backspace_count", "delete_count", "correction_count", "correction_rate", "correction_burst_count",
            
            # Pauses
            "pause_count", "pause_duration_mean", "pause_duration_max", "long_pause_count",
            
            # Rhythm
            "typing_rhythm_regularity", "burst_typing_score", "hesitation_score",
            
            # Overall
            "total_keys", "printable_keys", "modifier_usage", "is_active", "time_since_last_key",
        ]
    
    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self._key_timestamps.clear()
        self._key_durations.clear()
        self._inter_key_intervals.clear()
        self._pause_durations.clear()
        self._key_names.clear()
        self._key_timestamps.clear()
        self._last_key_time = None
        self._last_key_name = None
        self._pressed_keys.clear()
        self._total_keys = 0
        self._printable_keys = 0
        self._modifier_keys = 0
        self._recent_corrections.clear()
        self._correction_burst_count = 0
        self._in_correction_burst = False
        self._pause_durations.clear()
        self._in_pause = False
        self._pause_start_time = None