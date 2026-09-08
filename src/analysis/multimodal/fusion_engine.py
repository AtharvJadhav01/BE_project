"""
Multimodal fusion engine for human-state estimation.

Combines face, audio, and keyboard analysis using confidence-weighted
evidence fusion with cross-modal evidence detection.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from analysis.results import StateEstimate, AnalysisConfig
from analysis.interfaces import BaselineManager, BaselineStatus
from analysis.multimodal.alignment import TemporalAligner, ModalitySnapshot


@dataclass
class ModalityEvidence:
    """Evidence from a single modality for fusion."""
    modality: str
    state: str
    confidence: float
    uncertainty: float
    weight: float
    contributing_features: dict[str, float]
    data_quality: str
    is_fresh: bool
    is_stale: bool
    timestamp_monotonic: float
    baseline_ready: bool


@dataclass
class FusionResult:
    """Result of multimodal fusion."""
    state: str
    confidence: float
    uncertainty: float
    contributing_modalities: dict[str, float]  # modality -> weight
    contributing_features: dict[str, float]    # aggregated features
    modality_weights: dict[str, float]         # modality -> effective weight
    available_modalities: list[str]            # available modalities
    contributing_modalities_list: list[str]    # modalities that contributed
    evidence_summary: dict[str, any]           # detailed evidence
    modality_snapshots: dict[str, any]         # raw modality snapshots
    timestamp_monotonic: float
    wallclock: str
    data_quality: str
    baseline_ready: bool
    explanation: str = ""


class FusionEngine:
    """
    Multimodal fusion engine for human-state estimation.
    
    Combines face, audio, and keyboard analysis using:
    - Confidence-weighted evidence fusion
    - Cross-modal evidence agreement/disagreement detection
    - Missing modality handling
    - Temporal alignment
    - Configurable modality weights
    """
    
    def __init__(
        self,
        config: dict[str, any] | None = None,
        temporal_aligner: any = None,
        face_baseline: any = None,
        audio_baseline: any = None,
        keyboard_baseline: any = None,
    ):
        """
        Initialize fusion engine.
        
        Args:
            config: Fusion configuration
            temporal_aligner: TemporalAligner instance for temporal alignment
            face_baseline: Face baseline manager
            audio_baseline: Audio baseline manager
            keyboard_baseline: Keyboard baseline manager
        """
        self.config = config or {}
        
        # Temporal alignment
        self.temporal_aligner = temporal_aligner
        
        # Baseline managers for normalization
        self._baselines = {
            "face": face_baseline,
            "audio": audio_baseline,
            "keyboard": keyboard_baseline,
        }
        
        # Default modality weights (configurable)
        self.modality_weights = self.config.get("modality_weights", {
            "face": 1.0,
            "audio": 1.0,
            "keyboard": 1.0,
        })
        
        # Minimum modalities required for fusion
        self.min_modalities = self.config.get("min_modalities", 2)
        
        # Confidence thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.4)
        
        # Freshness and staleness
        self.freshness_thresholds = {
            "face": self.config.get("face_freshness_sec", 1.0),
            "audio": self.config.get("audio_freshness_sec", 1.0),
            "keyboard": self.config.get("keyboard_freshness_sec", 5.0),
        }
        self.stale_thresholds = {
            "face": self.config.get("face_stale_sec", 3.0),
            "audio": self.config.get("audio_stale_sec", 3.0),
            "keyboard": self.config.get("keyboard_stale_sec", 15.0),
        }
        
        # Smoothing
        self._last_fused_state: Optional[StateEstimate] = None
        self._state_history: list = []
        self._min_persistence_frames = self.config.get("min_persistence_frames", 3)
        self._state_persistence_count = 0
        
        # Smoothing factor
        self.smoothing_alpha = self.config.get("smoothing_alpha", 0.3)
        
        # Minimum modalities for fusion
        self.min_modalities_for_fusion = self.config.get("min_modalities_for_fusion", 2)
        
        # Debouncing
        self._last_state_change_time = 0.0
        self._debounce_sec = self.config.get("debounce_sec", 2.0)
        
        # Last fused state for smoothing
        self._last_fused_state: Optional[StateEstimate] = None
    
    def fuse(
        self,
        modality_snapshots: dict[str, any],  # modality -> ModalitySnapshot
        current_time: float | None = None,
    ) -> any:  # Returns fused StateEstimate
        """
        Fuse modality snapshots into a unified human state estimate.
        
        Args:
            modality_snapshots: Dict of modality -> ModalitySnapshot
            current_time: Current monotonic timestamp (uses time.monotonic() if None)
            
        Returns:
            Fused StateEstimate or None if insufficient data
        """
        if not current_time:
            current_time = time.monotonic()
        
        # Extract evidence from each modality
        evidence = self._extract_evidence(modality_snapshots)
        
        if not evidence:
            return self._create_uncertain_result("No modality data available")
        
        # Check minimum modalities
        fresh_evidence = [e for e in evidence if e.is_fresh and not e.is_stale]
        if len(fresh_evidence) < self.min_modalities_for_fusion:
            return self._create_uncertain_result(f"Insufficient fresh modalities: {len(fresh_evidence)}/{self.min_modalities_for_fusion}")
        
        # Compute modality weights
        modality_weights = self._compute_modality_weights(evidence)
        
        # Extract evidence for fusion
        evidence_items = []
        for modality, ev in evidence.items():
            if not ev.is_fresh or ev.is_stale:
                continue
            
            # Get normalized state weight
            weight = modality_weights.get(modality, 0.0)
            if weight <= 0:
                continue
            
            # Apply confidence and quality weighting
            effective_weight = self._compute_effective_weight(modality, evidence[modality], modality_weights)
            
            if effective_weight > 0:
                evidence_items.append({
                    "modality": modality,
                    "state": ev.state,
                    "confidence": ev.confidence,
                    "uncertainty": ev.uncertainty,
                    "weight": effective_weight,
                    "features": ev.contributing_features,
                })
        
        if not evidence_items:
            return self._create_uncertain_result("No valid evidence after weighting")
        
        # Fuse evidence
        fused_state = self._fuse_evidence(evidence_items)
        
        # Apply temporal smoothing
        fused_state = self._apply_temporal_smoothing(fused_state)
        
        # Check for state change debouncing
        fused_state = self._apply_debouncing(fused_state)
        
        # Update history
        self._update_history(fused_state)
        
        return fused_state
    
    def _extract_evidence(self, modality_snapshots: dict[str, any]) -> dict[str, any]:
        """Extract evidence from modality snapshots."""
        evidence = {}
        
        for modality, snapshot in modality_snapshots.items():
            if not snapshot or not snapshot.is_available:
                continue
            
            state_est = snapshot.state_estimate
            if not state_est:
                continue
            
            # Determine data quality
            data_quality = snapshot.data_quality
            is_fresh = snapshot.is_fresh
            is_stale = snapshot.is_stale
            
            # Check baseline readiness
            baseline_ready = snapshot.analysis_result.get("baseline_ready", False) if snapshot.analysis_result else False
            
            evidence = ModalityEvidence(
                modality=snapshot.modality,
                state=snapshot.state_estimate.state,
                confidence=snapshot.confidence,
                uncertainty=snapshot.uncertainty,
                weight=1.0,  # Will be computed later
                contributing_features=snapshot.contributing_features,
                data_quality=snapshot.data_quality,
                is_fresh=snapshot.is_fresh,
                is_stale=snapshot.is_stale,
                timestamp_monotonic=snapshot.timestamp_monotonic,
                baseline_ready=snapshot.analysis_result.get("baseline_ready", False) if snapshot.analysis_result else False,
            )
            
            evidence[snapshot.modality] = evidence
        
        return evidence
    
    def _compute_modality_weights(self, evidence: dict[str, any]) -> dict[str, float]:
        """Compute effective weights for each modality."""
        weights = {}
        
        for modality, ev in evidence.items():
            # Base weight from config
            base_weight = self.modality_weights.get(modality, 1.0)
            
            # Adjust for data quality
            quality_multiplier = {
                "good": 1.0,
                "degraded": 0.7,
                "poor": 0.3,
                "insufficient": 0.0,
            }.get(ev.data_quality, 0.5)
            
            # Adjust for freshness
            if not evidence[modality].is_fresh:
                freshness_penalty = 0.3
            else:
                freshness_penalty = 1.0
            
            # Adjust for staleness
            if evidence[modality].is_stale:
                staleness_penalty = 0.1
            else:
                staleness_penalty = 1.0
            
            # Baseline readiness
            baseline_ready = 1.0 if evidence[modality].baseline_ready else 0.5
            
            # Combine weights
            weight = (
                base_weight *
                quality_multiplier *
                freshness_penalty *
                staleness_penalty *
                baseline_ready
            )
            
            weights[modality] = max(0.0, weight)
        
        return weights
    
    def _compute_effective_weight(self, modality: str, evidence: any, weights: dict) -> float:
        """Compute effective weight for a modality's evidence."""
        base_weight = weights.get(modality, 0.0)
        if base_weight <= 0:
            return 0.0
        
        # Get evidence
        ev = evidence[modality]
        
        # Apply confidence scaling
        confidence_scale = evidence.confidence
        
        # Data quality multiplier
        quality_mult = {
            "good": 1.0,
            "degraded": 0.7,
            "poor": 0.3,
            "insufficient": 0.0,
        }.get(evidence.data_quality, 0.5)
        
        return base_weight * confidence_scale * quality_mult
    
    def _fuse_evidence(self, evidence_items: list[dict]) -> any:
        """Fuse evidence items into a single state estimate."""
        if not evidence_items:
            return self._create_uncertain_result("No valid evidence")
        
        # Group by state
        state_evidence = {}
        for item in evidence_items:
            state = item["state"]
            if state not in state_evidence:
                state_evidence[state] = []
            state_evidence[state].append(item)
        
        # Compute weighted scores for each state
        state_scores = {}
        state_features = {}
        state_modalities = {}
        
        for state, items in state_evidence.items():
            total_weight = 0.0
            weighted_confidence = 0.0
            total_features = {}
            contributing_modalities = []
            
            for item in items:
                weight = item["weight"]
                total_weight += weight
                weighted_confidence += item["confidence"] * weight
                
                # Aggregate contributing features
                for feat, val in item["features"].items():
                    if feat not in total_features:
                        total_features[feat] = []
                    total_features[feat].append(val * weight)
                
                contributing_modalities.append(item["modality"])
            
            if total_weight > 0:
                avg_confidence = weighted_confidence / total_weight
                state_scores[state] = avg_confidence * total_weight
                state_features[state] = {
                    k: sum(v) / sum([w for w in [i["weight"] for i in items]])
                    for k, v in total_features.items()
                }
                state_modalities[state] = list(set(contributing_modalities))
        
        if not state_scores:
            return self._create_uncertain_result("No valid state scores")
        
        # Select best state
        best_state = max(state_scores, key=state_scores.get)
        best_score = state_scores[best_state]
        
        # Compute confidence
        total_score = sum(state_scores.values())
        confidence = state_scores[best_state] / sum(state_scores.values()) if state_scores else 0.0
        confidence = min(0.95, max(0.05, confidence))
        
        # Uncertainty from competing states
        competing_scores = [s for s, v in state_scores.items() if v > 0]
        if len(competing_scores) > 1:
            max_score = max(competing_scores)
            second_max = sorted(competing_scores)[-2] if len(competing_scores) > 1 else 0
            margin = max_score - second_max
            uncertainty = 1.0 - min(1.0, margin / max_score) if max_score > 0 else 1.0
        else:
            uncertainty = 1.0 - confidence
        
        # Determine contributing modalities
        best_modalities = state_modalities.get(best_state, [])
        
        # Build explanation
        explanation = self._generate_explanation(best_state, state_evidence.get(best_state, []))
        
        # Create fused state estimate
        from analysis.results import StateEstimate
        from analysis.interfaces import DataQuality
        
        return StateEstimate(
            state=best_state,
            confidence=confidence,
            timestamp=time.monotonic(),
            wallclock=self._get_wallclock_timestamp(),
            contributing_features=state_features.get(best_state, {}),
            uncertainty=uncertainty,
            modality="multimodal",
            baseline_status=BaselineStatus.READY,
            raw_features={},
            normalized_features={},
        )
    
    def _generate_explanation(self, state: str, evidence: list[dict]) -> str:
        """Generate human-readable explanation for the fused state."""
        if not evidence:
            return f"State {state} determined with insufficient evidence."
        
        feature_summaries = []
        for item in evidence:
            modality = item["modality"]
            features = item.get("features", {})
            if features:
                top_features = sorted(
                    features.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:3]
                feat_str = ", ".join(f"{k}={v:.2f}" for k, v in top_features)
                feature_summaries.append(f"{modality}: {feat_str}")
        
        if feature_summaries:
            return f"State {state} indicated by: {'; '.join(feature_summaries)}"
        return f"State {state} determined from available evidence."
    
    def _apply_temporal_smoothing(self, fused_state: any) -> any:
        """Apply temporal smoothing to prevent rapid state switching."""
        if self._last_fused_state is None:
            self._last_fused_state = fused_state
            return fused_state
        
        # If state changed, check persistence
        if fused_state.state != self._last_fused_state.state:
            self._state_persistence_count = 1
            self._last_fused_state = fused_state
            return self._last_fused_state  # Keep previous state until persistence met
        
        # Same state - increment persistence
        self._state_persistence_count += 1
        
        # Apply confidence smoothing
        if self._last_fused_state:
            smoothed_confidence = (
                self.smoothing_alpha * fused_state.confidence +
                (1 - self.smoothing_alpha) * self._last_fused_state.confidence
            )
            fused_state.confidence = smoothed_confidence
        
        self._last_fused_state = fused_state
        return fused_state
    
    def _apply_debouncing(self, fused_state: any) -> any:
        """Apply debouncing to prevent rapid state changes."""
        current_time = time.monotonic()
        
        if self._last_fused_state and fused_state.state != self._last_fused_state.state:
            time_since_change = current_time - self._last_state_change_time
            if time_since_change < self._debounce_sec:
                # Not enough time passed, keep previous state
                return self._last_fused_state
        
        # Update change time
        if self._last_fused_state and fused_state.state != self._last_fused_state.state:
            self._last_state_change_time = time.monotonic()
        
        return fused_state
    
    def _update_history(self, fused_state: any) -> None:
        """Update state history for persistence tracking."""
        self._state_history.append({
            "state": fused_state.state,
            "confidence": fused_state.confidence,
            "timestamp": time.monotonic(),
        })
        
        # Keep limited history
        max_history = 100
        if len(self._state_history) > max_history:
            self._state_history = self._state_history[-max_history:]
    
    def _create_uncertain_result(self, reason: str) -> any:
        """Create an uncertain state estimate."""
        from analysis.results import StateEstimate
        from analysis.interfaces import BaselineStatus, DataQuality
        from datetime import datetime, timezone
        import time
        
        return StateEstimate(
            state="uncertain",
            confidence=0.0,
            timestamp=time.monotonic(),
            wallclock=datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            contributing_features={},
            uncertainty=1.0,
            modality="multimodal",
            baseline_status="not_started",
            raw_features={},
            normalized_features={},
        )
    
    def _get_wallclock_timestamp(self) -> str:
        """Get current wall-clock timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    def get_fusion_stats(self) -> dict:
        """Get fusion engine statistics."""
        return {
            "last_state": self._last_fused_state.state if self._last_fused_state else "none",
            "last_confidence": self._last_fused_state.confidence if self._last_fused_state else 0.0,
            "state_history_length": len(self._state_history),
            "persistence_count": getattr(self, '_state_persistence_count', 0),
            "debounce_active": time.monotonic() - getattr(self, '_last_state_change_time', 0) < getattr(self, '_debounce_sec', 2.0),
        }
    
    def reset(self) -> None:
        """Reset fusion engine state."""
        self._last_fused_state = None
        self._state_history.clear()
        self._state_persistence_count = 0
        self._last_state_change_time = 0.0
        self._last_fused_state = None