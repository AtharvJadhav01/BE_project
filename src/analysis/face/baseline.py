"""
Face baseline management for establishing personal facial feature baselines.

Extends the BaseBaselineManager to provide face-specific baseline functionality
for tracking personal norms of facial features during a session.
"""

from typing import Dict, Optional
from analysis.baseline import BaseBaselineManager


class FaceBaselineManager(BaseBaselineManager):
    """
    Personal baseline manager for facial features.
    
    Establishes and maintains personal baselines for facial features during
    a session to enable normalization and change detection.
    """
    
    def __init__(self, 
                 warmup_duration_sec: float = 10.0,
                 min_samples: int = 50,
                 ema_alpha: float = 0.1,
                 min_std: float = 1e-6):
        """
        Initialize face baseline manager.
        
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
        
        # Face-specific feature groups for organized baseline reporting
        self.feature_groups = {
            "eye": [
                "eye_openness_left", "eye_openness_right", "eye_openness_avg",
                "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate"
            ],
            "mouth": [
                "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry"
            ],
            "eyebrow": [
                "brow_inner_up", "brow_down_left", "brow_down_right",
                "brow_outer_up_left", "brow_outer_up_right"
            ],
            "head_pose": [
                "head_pitch", "head_yaw", "head_roll"
            ],
            "movement": [
                "facial_movement_velocity", "facial_symmetry"
            ],
            "expressivity": [
                "overall_expressivity"
            ]
        }
    
    def get_face_baseline_status(self) -> Dict[str, Dict[str, str]]:
        """
        Get baseline status organized by facial feature groups.
        
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
    
    def get_face_baseline_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get baseline statistics organized by facial feature groups.
        
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
    
    def is_face_baseline_ready(self, feature_group: Optional[str] = None) -> bool:
        """
        Check if facial feature baseline is ready.
        
        Args:
            feature_group: Optional specific group to check ("eye", "mouth", etc.)
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
    
    def get_normalized_face_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize raw facial features against personal baselines.
        
        Args:
            raw_features: Dictionary of raw facial feature values
            
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