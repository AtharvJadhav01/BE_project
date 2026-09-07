"""
Audio baseline management for establishing personal acoustic baselines.

Extends the BaseBaselineManager to provide audio-specific baseline functionality
for tracking personal norms of vocal features during a session.
"""

from typing import Dict, Optional
from analysis.baseline import BaseBaselineManager


class AudioBaselineManager(BaseBaselineManager):
    """
    Personal baseline manager for acoustic features.
    
    Establishes and maintains personal baselines for vocal features during
    a session to enable normalization and change detection.
    """
    
    def __init__(self, 
                 warmup_duration_sec: float = 10.0,
                 min_samples: int = 50,
                 ema_alpha: float = 0.1,
                 min_std: float = 1e-6):
        """
        Initialize audio baseline manager.
        
        Args:
            warmup_duration_sec: Duration to collect baseline before ready
            min_samples: Minimum samples required before baseline is ready
            ema_alpha: EMA decay factor (0 < alpha <= 1). Higher = faster adaptation.
            min_std: Minimum standard deviation to avoid division by zero
        """
        super().__init__(
            warmup_duration_sec=warmup_duration_sec,
            min_samples=min_samples,
            ema_alpha=ema_alpha,
            min_std=min_std
        )
        
        # Audio-specific feature groups for organized baseline reporting
        self.feature_groups = {
            "pitch": [
                "pitch_hz", "pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
                "pitch_valid", "pitch_confidence"
            ],
            "energy": [
                "rms_energy", "rms_energy_db", "energy_mean", "energy_std"
            ],
            "zcr": [
                "zcr", "zcr_mean", "zcr_std"
            ],
            "voice_activity": [
                "voice_activity", "voiced_ratio"
            ],
            "spectral": [
                "spectral_centroid", "spectral_rolloff", "spectral_flux"
            ],
        }
    
    def get_audio_baseline_status(self) -> Dict[str, Dict[str, str]]:
        """
        Get baseline status organized by audio feature groups.
        
        Returns:
            Dictionary mapping feature groups to their baseline statuses
        """
        status = {}
        all_baselines = self.get_all_baselines()
        
        for group_name, feature_names in self.feature_groups.items():
            group_status = {}
            for feature_name in feature_names:
                baseline = all_baselines.get(feature_name)
                if baseline:
                    group_status[feature_name] = baseline["status"]
                else:
                    group_status[feature_name] = "not_started"
            status[group_name] = group_status
            
        return status
    
    def get_audio_baseline_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get baseline statistics organized by audio feature groups.
        
        Returns:
            Dictionary mapping feature groups to their baseline statistics
        """
        summary = {}
        all_baselines = self.get_all_baselines()
        
        for group_name, feature_names in self.feature_groups.items():
            group_summary = {}
            for feature_name in feature_names:
                baseline = all_baselines.get(feature_name)
                if baseline:
                    group_summary[feature_name] = {
                        "mean": baseline["mean"],
                        "std": baseline["std"],
                        "count": baseline["count"],
                        "status": baseline["status"]
                    }
                else:
                    group_summary[feature_name] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "count": 0,
                        "status": "not_started"
                    }
            summary[group_name] = group_summary
            
        return summary
    
    def is_audio_baseline_ready(self, feature_group: Optional[str] = None) -> bool:
        """
        Check if audio feature baseline is ready.
        
        Args:
            feature_group: Optional specific group to check ("pitch", "energy", etc.)
                          If None, checks if ALL features are ready
                          
        Returns:
            True if baseline is ready for specified features
        """
        if feature_group is None:
            # Check if all features are ready
            return all(self.is_ready(feature) 
                      for group_features in self.feature_groups.values()
                      for feature in group_features)
        elif feature_group in self.feature_groups:
            # Check specific group
            return all(self.is_ready(feature) 
                      for feature in self.feature_groups[feature_group])
        else:
            return False
    
    def get_normalized_audio_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize raw audio features against personal baselines.
        
        Args:
            raw_features: Dictionary of raw audio feature values
            
        Returns:
            Dictionary of normalized (z-scored) feature values
        """
        normalized = {}
        
        for feature_name, value in raw_features.items():
            normalized_value = self.normalize(feature_name, value)
            if normalized_value is not None:
                normalized[feature_name] = normalized_value
            else:
                # If baseline not ready, return 0 (no change from baseline)
                normalized[feature_name] = 0.0
                
        return normalized


# Import Optional for type hints
from typing import Optional