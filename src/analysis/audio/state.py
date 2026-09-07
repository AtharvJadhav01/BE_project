"""
Audio state estimation for interpreting acoustic features into vocal states.

This module provides rule-based vocal state estimation that converts
normalized acoustic features into interpretable states.
"""

import math
from typing import Dict, List, Optional, Any

from analysis.interfaces import StateEstimator, StateEstimate, BaselineManager
from analysis.results import StateEstimate as StateEstimateModel


class AudioStateEstimator(StateEstimator):
    """
    Rule-based acoustic state estimator.
    
    Converts normalized acoustic features into interpretable states using
    observable prosodic features rather than claiming to detect internal
    emotional states.
    """
    
    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration thresholds for state detection
        self.thresholds = {
            # Pitch thresholds (relative to personal baseline - z-scores)
            "pitch_high_z": config.get("pitch_high_z", 1.5),        # Pitch significantly above baseline
            "pitch_low_z": config.get("pitch_low_z", -1.5),         # Pitch significantly below baseline
            "pitch_variability_high_z": config.get("pitch_variability_high_z", 1.5),  # High pitch variation
            "pitch_variability_low_z": config.get("pitch_variability_low_z", -1.0),   # Low pitch variation
            
            # Energy thresholds (relative to personal baseline - z-scores)
            "energy_high_z": config.get("energy_high_z", 1.0),      # Energy above baseline
            "energy_low_z": config.get("energy_low_z", -1.0),       # Energy below baseline
            "energy_variability_high_z": config.get("energy_variability_high_z", 1.0), # High energy variation
            
            # Voice activity thresholds
            "voice_activity_high": config.get("voice_activity_high", 0.7),  # High voice activity
            "voice_activity_low": config.get("voice_activity_low", 0.3),    # Low voice activity
            "voiced_ratio_high": config.get("voiced_ratio_high", 0.6),      # High voiced ratio
            
            # Pitch variability (absolute, not normalized)
            "pitch_std_high": config.get("pitch_std_high", 50.0),    # High pitch variation in Hz
            "pitch_std_low": config.get("pitch_std_low", 5.0),       # Low pitch variation
            
            # Energy variability
            "energy_std_high": config.get("energy_std_high", 10.0),   # High energy variation
            
            # Spectral
            "spectral_centroid_high": config.get("spectral_centroid_high", 3000.0), # Bright voice
        }
        
        # State definitions - ordered by priority
        self._state_definitions = [
            ("elevated_arousal_like_indicator", self._is_elevated_arousal),
            ("calm_like_indicator", self._is_calm),
            ("neutral", self._is_neutral),
        ]
    
    def estimate(self, features: Dict[str, float], 
                baseline_manager: BaselineManager) -> Optional[StateEstimate]:
        """
        Estimate acoustic state from normalized features.
        
        Args:
            features: Dictionary of normalized audio feature values (z-scores)
            baseline_manager: Baseline manager for this modality
            
        Returns:
            StateEstimate if estimation successful, None otherwise
        """
        if not features:
            return None
        
        # Check if baseline is ready for key audio features
        required_features = ["pitch_hz", "rms_energy_db", "pitch_std", "energy_std"]
        baseline_ready = all(baseline_manager.is_ready(f) for f in required_features)
        
        if not baseline_ready:
            return StateEstimateModel(
                state="uncertain",
                confidence=0.0,
                timestamp=time.monotonic(),
                wallclock=self._get_wallclock_timestamp(),
                contributing_features={},
                uncertainty=1.0,
                modality="audio",
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
                        modality="audio",
                        baseline_status=baseline_manager.get_status("pitch_hz") or "unknown",
                        raw_features=features.copy(),
                        normalized_features=self._get_normalized_features(features, baseline_manager)
                    )
            except Exception:
                continue
        
        # Fallback to neutral if no state matched
        return StateEstimateModel(
            state="neutral",
            confidence=0.5,
            timestamp=time.monotonic(),
            wallclock=self._get_wallclock_timestamp(),
            contributing_features={},
            uncertainty=0.5,
            modality="audio",
            baseline_status="ready",
            raw_features=features.copy()
        )
    
    def _get_normalized_features(self, features: Dict[str, float], baseline_manager: BaselineManager) -> Dict[str, float]:
        """Get normalized (z-scored) features."""
        normalized = {}
        for key, value in features.items():
            z = baseline_manager.normalize(key, value)
            if z is not None:
                normalized[key] = z
        return normalized
    
    def _get_wallclock_timestamp(self) -> str:
        """Get current wall-clock timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    def _is_elevated_arousal(self, features: Dict[str, float], baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for elevated arousal indicators: high pitch, high pitch variation, high energy, high voice activity."""
        try:
            # Get normalized values
            pitch_z = baseline_manager.normalize("pitch_hz", features.get("pitch_hz", 0))
            energy_z = baseline_manager.normalize("rms_energy_db", features.get("rms_energy_db", -60))
            pitch_std = features.get("pitch_std", 0)
            energy_std = features.get("energy_std", 0)
            voice_activity = features.get("voice_activity", 0)
            voiced_ratio = features.get("voiced_ratio", 0)
            spectral_centroid = features.get("spectral_centroid", 0)
            
            # Elevated arousal indicators
            indicators = []
            
            # High pitch relative to baseline
            if pitch_z is not None and pitch_z > self.thresholds["pitch_high_z"]:
                indicators.append(min(1.0, pitch_z / self.thresholds["pitch_high_z"]))
            
            # High energy relative to baseline
            if energy_z is not None and energy_z > self.thresholds["energy_high_z"]:
                indicators.append(min(1.0, energy_z / self.thresholds["energy_high_z"]))
            
            # High pitch variability
            if pitch_std > self.thresholds["pitch_std_high"]:
                indicators.append(min(1.0, pitch_std / self.thresholds["pitch_std_high"]))
            
            # High energy variability
            if energy_std > self.thresholds["energy_std_high"]:
                indicators.append(min(1.0, energy_std / self.thresholds["energy_std_high"]))
            
            # High voice activity
            if voice_activity > self.thresholds["voice_activity_high"]:
                indicators.append(min(1.0, voice_activity / self.thresholds["voice_activity_high"]))
            
            # High voiced ratio
            if voiced_ratio > self.thresholds["voiced_ratio_high"]:
                indicators.append(min(1.0, voiced_ratio / self.thresholds["voiced_ratio_high"]))
            
            # Bright spectral centroid (tense voice)
            if spectral_centroid > self.thresholds["spectral_centroid_high"]:
                indicators.append(min(1.0, spectral_centroid / self.thresholds["spectral_centroid_high"]))
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.95, max(0.1, confidence))
            
            contributing = {
                "pitch_z": pitch_z or 0,
                "energy_z": energy_z or 0,
                "pitch_std": pitch_std,
                "energy_std": energy_std,
                "voice_activity": voice_activity,
                "voiced_ratio": voiced_ratio,
                "spectral_centroid": spectral_centroid,
            }
            
            uncertainty = 1.0 - confidence
            
            return ("elevated_arousal_like_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_calm(self, features: Dict[str, float], baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for calm indicators: low pitch variation, low energy variation, low voice activity."""
        try:
            pitch_z = baseline_manager.normalize("pitch_hz", features.get("pitch_hz", 0))
            energy_z = baseline_manager.normalize("rms_energy_db", features.get("rms_energy_db", -60))
            pitch_std = features.get("pitch_std", 0)
            energy_std = features.get("energy_std", 0)
            voice_activity = features.get("voice_activity", 0)
            voiced_ratio = features.get("voiced_ratio", 0)
            
            # Calm indicators
            indicators = []
            
            # Low pitch variability
            if pitch_std < self.thresholds["pitch_std_low"]:
                indicators.append(min(1.0, (self.thresholds["pitch_std_low"] - pitch_std) / self.thresholds["pitch_std_low"]))
            
            # Low energy variability
            if energy_std < self.thresholds["energy_std_high"] * 0.3:
                indicators.append(1.0 - min(1.0, energy_std / (self.thresholds["energy_std_high"] * 0.3)))
            
            # Low voice activity
            if voice_activity < self.thresholds["voice_activity_low"]:
                indicators.append(min(1.0, (self.thresholds["voice_activity_low"] - voice_activity) / self.thresholds["voice_activity_low"]))
            
            # Low voiced ratio
            if voiced_ratio < self.thresholds["voiced_ratio_high"] * 0.5:
                indicators.append(1.0 - min(1.0, voiced_ratio / (self.thresholds["voiced_ratio_high"] * 0.5)))
            
            # Pitch and energy near baseline (within 0.5 z-score)
            if pitch_z is not None and abs(pitch_z) < 0.5:
                indicators.append(1.0 - abs(pitch_z) / 0.5)
            
            if energy_z is not None and abs(energy_z) < 0.5:
                indicators.append(1.0 - abs(energy_z) / 0.5)
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.95, max(0.1, confidence))
            
            contributing = {
                "pitch_std": pitch_std,
                "energy_std": energy_std,
                "voice_activity": voice_activity,
                "voiced_ratio": voiced_ratio,
                "pitch_z": pitch_z or 0,
                "energy_z": energy_z or 0,
            }
            
            uncertainty = 1.0 - confidence
            
            return ("calm_like_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_neutral(self, features: Dict[str, float], baseline_manager: BaselineManager) -> Optional[tuple]:
        """Check for neutral state: features near baseline, moderate variability."""
        try:
            pitch_z = baseline_manager.normalize("pitch_hz", features.get("pitch_hz", 0))
            energy_z = baseline_manager.normalize("rms_energy_db", features.get("rms_energy_db", -60))
            pitch_std = features.get("pitch_std", 0)
            energy_std = features.get("energy_std", 0)
            voice_activity = features.get("voice_activity", 0)
            
            # Neutral indicators - features within normal ranges
            indicators = []
            
            # Pitch near baseline
            if pitch_z is not None:
                indicators.append(max(0.0, 1.0 - abs(pitch_z) / 1.5))
            
            # Energy near baseline
            if energy_z is not None:
                indicators.append(max(0.0, 1.0 - abs(energy_z) / 1.5))
            
            # Moderate pitch variability
            if 10.0 <= pitch_std <= 40.0:
                indicators.append(1.0)
            elif pitch_std < 10.0:
                indicators.append(0.5)
            else:
                indicators.append(0.3)
            
            # Moderate energy variability
            if 2.0 <= energy_std <= 8.0:
                indicators.append(1.0)
            elif energy_std < 2.0:
                indicators.append(0.5)
            else:
                indicators.append(0.3)
            
            # Moderate voice activity
            if 0.3 <= voice_activity <= 0.7:
                indicators.append(1.0)
            else:
                indicators.append(0.5)
            
            if not indicators:
                return None
            
            confidence = sum(indicators) / len(indicators)
            confidence = min(0.9, max(0.3, confidence))
            
            contributing = {
                "pitch_z": pitch_z or 0,
                "energy_z": energy_z or 0,
                "pitch_std": pitch_std,
                "energy_std": energy_std,
                "voice_activity": voice_activity,
            }
            
            uncertainty = 1.0 - confidence
            
            return ("neutral", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def get_possible_states(self) -> List[str]:
        """Return list of possible state labels."""
        return [
            "elevated_arousal_like_indicator",
            "calm_like_indicator",
            "neutral",
            "uncertain"
        ]
    
    def reset(self) -> None:
        """Reset internal state."""
        pass  # No internal state to reset


# Import time at module level
import time
from datetime import datetime, timezone
from analysis.interfaces import BaselineStatus