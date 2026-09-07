"""
Keyboard state estimation for interpreting typing behavior into states.

This module provides rule-based typing state estimation that converts
normalized keyboard features into interpretable states.
"""

import time
from typing import Dict, List, Optional, Any

from analysis.interfaces import StateEstimator, StateEstimate, BaselineManager
from analysis.results import StateEstimate as StateEstimateModel
from analysis.interfaces import BaselineStatus


class KeyboardStateEstimator(StateEstimator):
    """
    Rule-based typing state estimator.
    
    Converts normalized keyboard features into interpretable states using
    observable typing behavior patterns rather than claiming to detect
    internal psychological states.
    """
    
    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration thresholds for state detection
        self.thresholds = {
            # Typing speed thresholds (relative to baseline z-scores)
            "typing_speed_high_z": config.get("typing_speed_high_z", 1.5),    # Fast typing
            "typing_speed_low_z": config.get("typing_speed_low_z", -1.5),     # Slow typing
            
            # Inter-key interval variability
            "iki_cv_high_z": config.get("iki_cv_high_z", 1.5),               # Irregular rhythm
            "iki_cv_low_z": config.get("iki_cv_low_z", -1.0),                # Very regular
            
            # Correction thresholds
            "correction_rate_high_z": config.get("correction_rate_high_z", 1.5),  # High corrections
            "correction_burst_high": config.get("correction_burst_high", 3),       # Correction bursts
            
            # Pause thresholds
            "long_pause_high_z": config.get("long_pause_high_z", 1.5),       # Many long pauses
            "pause_duration_high_z": config.get("pause_duration_high_z", 1.5),  # Long pauses
            
            # Rhythm thresholds
            "rhythm_irregular_z": config.get("rhythm_irregular_z", 1.5),     # Irregular rhythm
            "burst_typing_high": config.get("burst_typing_high", 0.6),       # Burst typing
            "hesitation_high": config.get("hesitation_high", 0.5),           # High hesitation
            
            # Key duration
            "key_duration_high_z": config.get("key_duration_high_z", 1.5),   # Long key holds
            
            # Typing speed (absolute)
            "typing_speed_fast_kpm": config.get("typing_speed_fast_kpm", 300),  # Very fast
            "typing_speed_slow_kpm": config.get("typing_speed_slow_kpm", 30),   # Very slow
        }
        
        # State definitions - ordered by priority (higher priority checked first)
        self._state_definitions = [
            ("elevated_typing_tension_indicator", self._is_elevated_tension),
            ("hesitation_indicator", self._is_hesitation),
            ("focused_typing_indicator", self._is_focused_typing),
            ("neutral_typing_indicator", self._is_neutral_typing),
        ]
    
    def estimate(self, features: Dict[str, float], 
                baseline_manager: BaselineManager) -> Optional[StateEstimate]:
        """
        Estimate typing state from normalized features.
        
        Args:
            features: Dictionary of normalized keyboard feature values (z-scores)
            baseline_manager: Baseline manager for this modality
            
        Returns:
            StateEstimate if estimation successful, None otherwise
        """
        if not features:
            return None
        
        # Check if baseline is ready for key features
        required_features = ["typing_speed_kpm", "iki_mean", "iki_cv", "correction_rate"]
        baseline_ready = all(baseline_manager.is_ready(f) for f in required_features)
        
        if not baseline_ready:
            return StateEstimateModel(
                state="uncertain",
                confidence=0.0,
                timestamp=time.monotonic(),
                wallclock=self._get_wallclock_timestamp(),
                contributing_features={},
                uncertainty=1.0,
                modality="keyboard",
                baseline_status=BaselineStatus.WARMING_UP,
                raw_features=features.copy()
            )
        
        # Evaluate each state in priority order
        for state_name, evaluation_func in self._state_definitions:
            try:
                result = evaluation_func(features, baseline_manager)
                if result is not None:
                    state, confidence, contributing_features, uncertainty = result
                    
                    return StateEstimateModel(
                        state=state,
                        confidence=confidence,
                        timestamp=time.monotonic(),
                        wallclock=self._get_wallclock_timestamp(),
                        contributing_features=contributing_features,
                        uncertainty=uncertainty,
                        modality="keyboard",
                        baseline_status=baseline_manager.get_status("typing_speed_kpm") or "unknown",
                        raw_features=features.copy()
                    )
            except Exception:
                continue
        
        # Fallback to neutral if no state matched
        return StateEstimateModel(
            state="neutral_typing_indicator",
            confidence=0.5,
            timestamp=time.monotonic(),
            wallclock=self._get_wallclock_timestamp(),
            contributing_features={},
            uncertainty=0.5,
            modality="keyboard",
            baseline_status="ready",
            raw_features=features.copy()
        )
    
    def _is_elevated_tension(self, features: Dict[str, float], 
                            baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for elevated typing tension: high corrections, irregular rhythm, long pauses."""
        try:
            # Elevated tension indicators
            indicators = []
            
            # High correction rate
            correction_rate_z = self._get_z_score(features, "correction_rate", baseline_manager)
            if correction_rate_z is not None and correction_rate_z > self.thresholds["correction_rate_high_z"]:
                indicators.append(min(1.0, correction_rate_z / self.thresholds["correction_rate_high_z"]))
            
            # Correction bursts
            correction_burst = features.get("correction_burst_count", 0)
            if correction_burst >= self.thresholds["correction_burst_high"]:
                indicators.append(min(1.0, correction_burst / 5.0))
            
            # Irregular rhythm
            rhythm = features.get("typing_rhythm_regularity", 1.0)
            if rhythm < 0.5:  # Low regularity = irregular
                indicators.append(1.0 - rhythm)
            
            # Long pauses
            long_pause_z = self._get_z_score(features, "long_pause_count", baseline_manager)
            if long_pause_z is not None and long_pause_z > self.thresholds["long_pause_high_z"]:
                indicators.append(min(1.0, long_pause_z / self.thresholds["long_pause_high_z"]))
            
            # High hesitation
            hesitation = features.get("hesitation_score", 0.0)
            if hesitation > self.thresholds["hesitation_high"]:
                indicators.append(min(1.0, hesitation / self.thresholds["hesitation_high"]))
            
            # High pause duration
            pause_dur_z = self._get_z_score(features, "pause_duration_mean", baseline_manager)
            if pause_dur_z is not None and pause_dur_z > self.thresholds["pause_duration_high_z"]:
                indicators.append(min(1.0, pause_dur_z / self.thresholds["pause_duration_high_z"]))
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.95, max(0.1, confidence))
            
            contributing = {
                "correction_rate_z": correction_rate_z or 0,
                "correction_burst_count": correction_burst,
                "typing_rhythm_regularity": rhythm,
                "long_pause_count_z": long_pause_z or 0,
                "hesitation_score": features.get("hesitation_score", 0.0),
                "pause_duration_mean_z": pause_dur_z or 0,
            }
            
            uncertainty = 1.0 - confidence
            
            return ("elevated_typing_tension_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_hesitation(self, features: Dict[str, float], 
                      baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for hesitation: long IKIs, many pauses, irregular rhythm."""
        try:
            indicators = []
            
            # Long inter-key intervals
            iki_mean_z = self._get_z_score(features, "iki_mean", baseline_manager)
            if iki_mean_z is not None and iki_mean_z > 1.0:
                indicators.append(min(1.0, iki_mean_z / 2.0))
            
            # Many pauses
            pause_count = features.get("pause_count", 0)
            total_keys = features.get("total_keys", 1)
            if total_keys > 0 and pause_count / total_keys > 0.1:
                indicators.append(min(1.0, (pause_count / total_keys) / 0.2))
            
            # Long pauses
            pause_dur_z = self._get_z_score(features, "pause_duration_mean", baseline_manager)
            if pause_dur_z is not None and pause_dur_z > 1.0:
                indicators.append(min(1.0, pause_dur_z / 2.0))
            
            # Irregular rhythm
            rhythm = features.get("typing_rhythm_regularity", 1.0)
            if rhythm < 0.4:
                indicators.append(1.0 - rhythm)
            
            # High hesitation score
            hesitation = features.get("hesitation_score", 0.0)
            if hesitation > 0.3:
                indicators.append(min(1.0, hesitation / 0.5))
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.9, max(0.1, confidence))
            
            contributing = {
                "iki_mean_z": iki_mean_z or 0,
                "pause_count": pause_count,
                "pause_duration_mean_z": self._get_z_score(features, "pause_duration_mean", baseline_manager) or 0,
                "typing_rhythm_regularity": rhythm,
                "hesitation_score": features.get("hesitation_score", 0.0),
            }
            
            uncertainty = 1.0 - confidence
            
            return ("hesitation_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_focused_typing(self, features: Dict[str, float], 
                          baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for focused typing: fast, regular, low corrections, few pauses."""
        try:
            indicators = []
            
            # Fast typing speed
            typing_speed = features.get("typing_speed_kpm", 0)
            if typing_speed > self.thresholds["typing_speed_fast_kpm"]:
                indicators.append(min(1.0, typing_speed / self.thresholds["typing_speed_fast_kpm"]))
            
            # Regular rhythm
            rhythm = features.get("typing_rhythm_regularity", 0)
            if rhythm > 0.7:
                indicators.append(rhythm)
            
            # Low correction rate
            correction_rate = features.get("correction_rate", 0)
            if correction_rate < 2.0:  # Less than 2 corrections per minute
                indicators.append(1.0 - min(1.0, correction_rate / 2.0))
            
            # Few pauses
            pause_count = features.get("pause_count", 0)
            total_keys = features.get("total_keys", 1)
            if total_keys > 0 and pause_count / total_keys < 0.05:
                indicators.append(1.0 - (pause_count / total_keys) / 0.05)
            
            # Regular IKI (low CV)
            iki_cv = features.get("iki_cv", 1.0)
            if iki_cv < 0.3:
                indicators.append(1.0 - iki_cv / 0.3)
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.9, max(0.1, confidence))
            
            contributing = {
                "typing_speed_kpm": typing_speed,
                "typing_rhythm_regularity": features.get("typing_rhythm_regularity", 0),
                "correction_rate": correction_rate,
                "pause_ratio": pause_count / max(1, total_keys),
                "iki_cv": iki_cv,
            }
            
            uncertainty = 1.0 - confidence
            
            return ("focused_typing_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_neutral_typing(self, features: Dict[str, float], 
                          baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for neutral typing: features near baseline, moderate everything."""
        try:
            indicators = []
            
            # Speed near baseline
            speed_z = self._get_z_score(features, "typing_speed_kpm", baseline_manager)
            if speed_z is not None:
                indicators.append(max(0.0, 1.0 - abs(speed_z) / 1.5))
            
            # IKI near baseline
            iki_z = self._get_z_score(features, "iki_mean", baseline_manager)
            if iki_z is not None:
                indicators.append(max(0.0, 1.0 - abs(iki_z) / 1.5))
            
            # Moderate rhythm
            rhythm = features.get("typing_rhythm_regularity", 0.5)
            if 0.4 <= rhythm <= 0.8:
                indicators.append(1.0 - abs(rhythm - 0.6) / 0.2)
            
            # Moderate corrections
            correction_rate = features.get("correction_rate", 0)
            if 0.5 <= correction_rate <= 5.0:
                indicators.append(1.0)
            elif correction_rate < 0.5:
                indicators.append(0.5 + correction_rate)
            else:
                indicators.append(max(0.0, 1.0 - (correction_rate - 5.0) / 5.0))
            
            # Moderate pauses
            pause_ratio = features.get("pause_count", 0) / max(1, features.get("total_keys", 1))
            if pause_ratio < 0.1:
                indicators.append(1.0 - pause_ratio / 0.1)
            
            # Moderate rhythm
            rhythm = features.get("typing_rhythm_regularity", 0)
            if rhythm > 0.5:
                indicators.append(rhythm)
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.85, max(0.3, confidence))
            
            contributing = {
                "typing_speed_kpm": features.get("typing_speed_kpm", 0),
                "typing_speed_z": speed_z or 0,
                "iki_mean_z": self._get_z_score(features, "iki_mean", baseline_manager) or 0,
                "typing_rhythm_regularity": rhythm,
                "correction_rate": features.get("correction_rate", 0),
                "pause_ratio": features.get("pause_count", 0) / max(1, features.get("total_keys", 1)),
            }
            
            uncertainty = 1.0 - confidence
            
            return ("neutral_typing_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _get_z_score(self, features: Dict[str, float], feature_name: str, 
                    baseline_manager: BaselineManager) -> Optional[float]:
        """Get z-score for a feature from baseline manager."""
        value = features.get(feature_name)
        if value is None:
            return None
        return baseline_manager.normalize(feature_name, value)
    
    def _get_wallclock_timestamp(self) -> str:
        """Get current wall-clock timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    def get_possible_states(self) -> List[str]:
        """Return list of possible state labels."""
        return [
            "elevated_typing_tension_indicator",
            "hesitation_indicator",
            "focused_typing_indicator",
            "neutral_typing_indicator",
            "uncertain"
        ]
    
    def reset(self) -> None:
        """Reset internal state."""
        pass