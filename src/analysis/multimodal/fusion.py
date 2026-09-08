"""
Multimodal fusion module - main entry point.

Combines temporal alignment, evidence extraction, and confidence-weighted
fusion to produce a unified human-state estimate from face, audio, and keyboard.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from analysis.interfaces import BaselineManager, BaselineStatus, DataQuality
from analysis.results import StateEstimate, StateEstimate as StateEstimateModel
from analysis.interfaces import BaselineStatus

from analysis.multimodal.alignment import TemporalAligner, ModalitySnapshot
from analysis.multimodal.fusion_engine import FusionEngine


@dataclass
class HumanStateEstimate:
    """Complete human state estimate with all metadata."""
    state: str
    confidence: float
    uncertainty: float
    timestamp: float
    wallclock: str
    contributing_modalities: dict[str, float]
    contributing_features: dict[str, float]
    modality_weights: dict[str, float]
    available_modalities: list[str]
    contributing_modalities: list[str]
    modality_snapshots: dict[str, any]
    baseline_ready: bool
    data_quality: str
    explanation: str


class HumanStateEstimator:
    """
    Top-level human state estimator combining face, audio, and keyboard analysis.
    
    Orchestrates:
    1. Temporal alignment of modality results
    2. Evidence extraction from each modality
    3. Confidence-weighted evidence fusion
    4. Cross-modal evidence agreement detection
    5. Temporal smoothing and debouncing
    8. Explanation generation
    """
    
    def __init__(
        self,
        config: dict[str, any] | None = None,
        face_baseline: any = None,
        audio_baseline: any = None,
        keyboard_baseline: any = None,
        temporal_aligner: any = None,
    ):
        """
        Initialize human state estimator.
        
        Args:
            config: Fusion configuration
            face_baseline: Face baseline manager
            audio_baseline: Audio baseline manager
            keyboard_baseline: Keyboard baseline manager
            temporal_aligner: Pre-configured temporal aligner (optional)
        """
        self.config = config or {}
        
        # Temporal alignment
        self.temporal_aligner = temporal_aligner or TemporalAligner(
            freshness_thresholds=self.config.get("freshness_thresholds", {}),
            stale_thresholds=self.config.get("stale_thresholds", {}),
        )
        
        # Fusion engine
        self.fusion_engine = FusionEngine(
            config=self.config.get("fusion", {}),
            temporal_aligner=None,  # We handle alignment externally
            face_baseline=None,  # Will be set via modality baselines
            audio_baseline=None,
            keyboard_baseline=None,
        )
        
        # Store baseline managers for external access
        self._face_baseline = None
        self._audio_baseline = None
        self._keyboard_baseline = None
        
        # Session state
        self._session_active = False
        self._last_update_time = 0.0
        self._min_update_interval = self.config.get("update_interval_ms", 500) / 1000.0
        
        # Statistics
        self._fusion_count = 0
        self._last_fusion_time = 0.0
    
    def update(
        self,
        face_result: any = None,
        audio_result: any = None,
        keyboard_result: any = None,
        current_time: float | None = None,
    ) -> Optional[dict]:
        """
        Update with new modality results and produce fused state estimate.
        
        Args:
            face_result: Face analysis result (AnalysisResult or None)
            audio_result: Audio analysis result (AnalysisResult or None)
            keyboard_result: Keyboard analysis result (AnalysisResult or None)
            current_time: Current monotonic timestamp (uses time.monotonic() if None)
            
        Returns:
            Fused human state estimate dict or None if insufficient data
        """
        current_time = current_time or time.monotonic()
        
        # Rate limiting
        if current_time - self._last_update_time < self._min_update_interval:
            return None
        
        # Create modality snapshots
        snapshots = self._create_snapshots(face_result, audio_result, keyboard_result)
        
        # Add to temporal aligner
        for modality, snapshot in snapshots.items():
            if snapshot:
                self.temporal_aligner.add_snapshot(snapshot)
        
        # Get aligned snapshots
        aligned = self.temporal_aligner.get_aligned_snapshots()
        
        # Check if we have enough data
        available = self.temporal_aligner.get_available_modalities()
        if len(available) < 2:  # min_modalities default is 2
            return self._create_uncertain_result("Insufficient modalities")
        
        # Fuse
        fused_state = self.fusion_engine.fuse(
            modality_snapshots=snapshots,
            current_time=time.monotonic(),
        )
        
        if not result:
            return self._create_uncertain_result("Fusion returned no result")
        
        # Build output
        return self._build_output(fused_state, current_time)
    
    def _create_snapshots(
        self,
        face_result: any,
        audio_result: any,
        keyboard_result: any,
    ) -> dict:
        """Create modality snapshots from analysis results."""
        snapshots = {}
        
        for modality, result in [
            ("face", face_result),
            ("audio", audio_result),
            ("keyboard", keyboard_result),
        ]:
            if not result:
                continue
            
            # Extract state estimate
            state_est = None
            if hasattr(result, 'state_estimate') and result.state_estimate:
                state_est = result.state_estimate
            elif hasattr(result, 'state_estimate') and result.state_estimate:
                state_est = result.state_estimate
            
            # Get feature vector
            feature_vec = None
            if hasattr(result, 'feature_vector') and result.feature_vector:
                feature_vec = result.feature_vector
            
            if not feature_vec:
                continue
            
            # Build snapshot
            snapshot = ModalitySnapshot(
                modality=modality,
                timestamp_monotonic=result.timestamp if hasattr(result, 'timestamp') else time.monotonic(),
                wallclock=result.wallclock if hasattr(result, 'wallclock') else "",
                analysis_result=result.to_dict() if hasattr(result, 'to_dict') else {},
                state_estimate=result.state_estimate if hasattr(result, 'state_estimate') else None,
                is_available=True,
                data_quality=result.feature_vector.quality.value if hasattr(result, 'feature_vector') and result.feature_vector else "insufficient",
            )
            
            snapshots[result.modality if hasattr(result, 'modality') else 'unknown'] = snapshot
        
        return snapshots
    
    def _build_output(self, fused_state: any, current_time: float) -> dict:
        """Build output dictionary from fused state."""
        modality_snapshots = self.temporal_aligner.get_aligned_snapshots()
        
        return {
            "state": fused_state.state,
            "confidence": fused_state.confidence,
            "uncertainty": fused_state.uncertainty,
            "timestamp": fused_state.timestamp,
            "wallclock": fused_state.wallclock,
            "contributing_modalities": {
                m: self._get_modality_weight(m) 
                for m in self.fusion_engine.get_fusion_stats().get("contributing_modalities", [])
            },
            "contributing_features": fused_state.contributing_features,
            "modality_weights": self._get_modality_weights(),
            "available_modalities": self.temporal_aligner.get_available_modalities(),
            "contributing_modalities": self._get_contributing_modalities(),
            "modality_snapshots": {
                m: s.to_dict() if hasattr(s, 'to_dict') else {}
                for m, s in self.temporal_aligner.get_aligned_snapshots().items()
            },
            "baseline_ready": all(
                self._get_baseline(m).is_ready() if self._get_baseline(m) else False
                for m in ["face", "audio", "keyboard"]
            ),
            "data_quality": "good",
            "explanation": self._generate_explanation(),
        }
    
    def _generate_explanation(self) -> str:
        """Generate human-readable explanation of the fused state."""
        return "Multimodal fusion based on available modalities."
    
    def _get_modality_weight(self, modality: str) -> float:
        return self.fusion_engine.modality_weights.get(modality, 0.0)
    
    def _get_modality_weights(self) -> dict[str, float]:
        return self.fusion_engine.modality_weights.copy()
    
    def _get_contributing_modalities(self) -> list[str]:
        stats = self.fusion_engine.get_fusion_stats()
        return stats.get("contributing_modalities", [])
    
    def _create_uncertain_result(self, reason: str) -> dict:
        """Create an uncertain result."""
        from datetime import datetime, timezone
        return {
            "state": "uncertain",
            "confidence": 0.0,
            "uncertainty": 1.0,
            "timestamp": time.monotonic(),
            "wallclock": datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            "contributing_modalities": {},
            "contributing_features": {},
            "modality_weights": {},
            "available_modalities": [],
            "contributing_modalities": [],
            "modality_snapshots": {},
            "baseline_ready": False,
            "data_quality": "insufficient",
            "explanation": reason,
        }
    
    def update_modality_result(self, modality: str, result: any) -> None:
        """Update a modality result through the temporal aligner."""
        # This is called from modality callbacks
        pass
    
    def set_modality_availability(self, modality: str, available: bool) -> None:
        """Set modality availability."""
        self.temporal_aligner.update_availability(modality, available)
    
    def get_modality_status(self) -> dict[str, dict]:
        """Get status of all modalities."""
        return self.temporal_aligner.get_modality_status()
    
    def get_fusion_stats(self) -> dict:
        """Get fusion statistics."""
        return {
            "fusion_count": self._fusion_count,
            "last_fusion_time": self._last_fusion_time,
            "last_update": self._last_update_time,
            "min_update_interval": self._min_update_interval,
            "fusion_engine_stats": self.fusion_engine.get_fusion_stats(),
            "temporal_aligner_status": self.temporal_aligner.get_modality_status(),
        }
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.temporal_aligner.reset()
        self.fusion_engine.reset()
        self._fusion_count = 0
        self._last_fusion_time = 0.0
        self._last_update_time = 0.0
    
    def update_config(self, config: dict) -> None:
        """Update configuration."""
        self.config.update(config)
        self.fusion_engine.config.update(config.get("fusion", {}))
        self.temporal_aligner.freshness_thresholds.update(
            self.config.get("freshness_thresholds", {})
        )
        self.temporal_aligner.stale_thresholds.update(
            self.config.get("stale_thresholds", {})
        )