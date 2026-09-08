"""
Temporal alignment for multimodal fusion.

Handles alignment of face, audio, and keyboard analysis results
across different update rates and timestamps.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from analysis.results import AnalysisResult, StateEstimate


@dataclass
class ModalitySnapshot:
    """Single modality analysis snapshot with metadata."""
    modality: str                    # "face", "audio", "keyboard"
    timestamp_monotonic: float       # monotonic timestamp (seconds since session start)
    wallclock: str                   # ISO 8601 UTC
    analysis_result: Any             # AnalysisResult object
    state_estimate: Any              # StateEstimate object
    is_fresh: bool = False           # within freshness threshold
    is_stale: bool = False           # exceeds freshness threshold
    is_available: bool = True        # modality is active/capturing
    data_quality: str = "good"       # "good", "degraded", "poor", "insufficient"
    
    @property
    def state(self) -> str:
        """Get state label if available."""
        if self.state_estimate:
            return self.state_estimate.state
        return "unknown"
    
    @property
    def confidence(self) -> float:
        """Get confidence if available."""
        if self.state_estimate:
            return self.state_estimate.confidence
        return 0.0
    
    @property
    def uncertainty(self) -> float:
        """Get uncertainty if available."""
        if self.state_estimate:
            return self.state_estimate.uncertainty
        return 1.0
    
    @property
    def contributing_features(self) -> dict[str, float]:
        """Get contributing features if available."""
        if self.state_estimate:
            return self.state_estimate.contributing_features
        return {}


class TemporalAligner:
    """
    Aligns multimodal analysis results across different update rates.
    
    Handles:
    - Different update rates (face ~15fps, audio ~100ms chunks, keyboard event-driven)
    - Timestamp synchronization using monotonic clock
    - Freshness/staleness detection
    - Missing modality handling
    - Configurable freshness thresholds
    """
    
    def __init__(
        self,
        freshness_thresholds: Optional[Dict[str, float]] = None,
        stale_thresholds: Optional[Dict[str, float]] = None,
        max_history: int = 100
    ):
        """
        Initialize temporal aligner.
        
        Args:
            freshness_thresholds: Max age (seconds) for fresh data per modality
            stale_thresholds: Max age (seconds) before data is stale per modality
            max_history: Maximum snapshots to keep per modality
        """
        self.max_history = max_history
        
        # Default freshness thresholds (seconds) - configurable per modality
        self.freshness_thresholds = freshness_thresholds or {
            "face": 1.0,      # 1 second for face (15fps = ~67ms per frame)
            "audio": 1.0,     # 1 second for audio (100ms chunks)
            "keyboard": 5.0,  # 5 seconds for keyboard (event-driven, bursty)
        }
        
        self.stale_thresholds = stale_thresholds or {
            "face": 3.0,      # 3 seconds stale for face
            "audio": 3.0,     # 3 seconds stale for audio
            "keyboard": 15.0, # 15 seconds stale for keyboard (bursty)
        }
        
        # Per-modality history buffers
        self._history: Dict[str, List] = {
            "face": [],
            "audio": [],
            "keyboard": [],
        }
        
        # Latest snapshots
        self._latest: Dict[str, Any] = {}
        
        # Modality availability tracking
        self._availability: Dict[str, bool] = {
            "face": True,
            "audio": True,
            "keyboard": True,
        }
    
    def add_snapshot(self, snapshot: Any) -> None:
        """Add a modality snapshot to the alignment buffer."""
        modality = snapshot.modality
        
        if modality not in self._history:
            self._history[modality] = []
            self._latest[modality] = None
            self._availability[modality] = True
        
        self._history[modality].append(snapshot)
        self._latest[modality] = snapshot
        
        # Trim history
        if len(self._history[modality]) > self.max_history:
            self._history[modality] = self._history[modality][-self.max_history:]
    
    def update_availability(self, modality: str, available: bool) -> None:
        """Update modality availability status."""
        self._availability[modality] = available
        if not available and modality in self._latest:
            # Mark existing snapshot as unavailable
            self._latest[modality].is_available = False
    
    def get_latest(self, modality: str, current_time: Optional[float] = None) -> Optional[Any]:
        """Get the latest snapshot for a modality, updating freshness flags."""
        if current_time is None:
            current_time = time.monotonic()
        
        snapshot = self._latest.get(modality)
        if not snapshot:
            return None
        
        # Update freshness/staleness flags
        age = current_time - snapshot.timestamp_monotonic
        freshness_threshold = self.freshness_thresholds.get(modality, 1.0)
        stale_threshold = self.stale_thresholds.get(modality, 3.0)
        
        snapshot.is_fresh = age <= freshness_threshold
        snapshot.is_stale = age > stale_threshold
        
        return snapshot
    
    def get_aligned_snapshots(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Get latest snapshots from all available modalities with freshness flags.
        
        Returns:
            Dict of modality -> snapshot (or None if unavailable)
        """
        if current_time is None:
            current_time = time.monotonic()
        
        snapshots = {}
        for modality in ["face", "audio", "keyboard"]:
            snapshot = self.get_latest(modality, current_time)
            if snapshot and snapshot.is_available:
                snapshots[modality] = snapshot
        
        return snapshots
    
    def get_modality_status(self) -> Dict[str, Dict]:
        """Get status of all modalities."""
        current_time = time.monotonic()
        status = {}
        for modality in ["face", "audio", "keyboard"]:
            snapshot = self._latest.get(modality)
            if snapshot:
                age = current_time - snapshot.timestamp_monotonic
                status[modality] = {
                    "available": snapshot.is_available,
                    "fresh": snapshot.is_fresh,
                    "stale": snapshot.is_stale,
                    "age_sec": age,
                    "state": snapshot.state,
                    "confidence": snapshot.confidence,
                    "data_quality": snapshot.data_quality,
                }
            else:
                status[modality] = {
                    "available": self._availability.get(modality, False),
                    "fresh": False,
                    "stale": True,
                    "age_sec": None,
                    "state": "unknown",
                    "confidence": 0.0,
                    "data_quality": "insufficient",
                }
        return status
    
    def get_available_modalities(self) -> List[str]:
        """Get list of available and fresh modalities."""
        current_time = time.monotonic()
        available = []
        for modality in ["face", "audio", "keyboard"]:
            snapshot = self._latest.get(modality)
            if snapshot and snapshot.is_available and snapshot.is_fresh:
                available.append(modality)
        return available
    
    def get_primary_snapshot(self, current_time: Optional[float] = None) -> Optional[Tuple[str, Any]]:
        """
        Get the most recent snapshot across all modalities.
        
        Returns:
            Tuple of (modality, snapshot) or None if no data
        """
        if current_time is None:
            current_time = time.monotonic()
        
        latest_snapshot = None
        latest_modality = None
        latest_time = -1
        
        for modality in ["face", "audio", "keyboard"]:
            snapshot = self._latest.get(modality)
            if snapshot and snapshot.is_available:
                if snapshot.timestamp_monotonic > latest_time:
                    latest_time = snapshot.timestamp_monotonic
                    latest_snapshot = snapshot
                    latest_modality = modality
        
        if latest_snapshot:
            return (latest_modality, latest_snapshot)
        return None
    
    def reset(self) -> None:
        """Reset all alignment state."""
        for modality in self._history:
            self._history[modality].clear()
        self._latest.clear()
        self._availability = {
            "face": True,
            "audio": True,
            "keyboard": True,
        }


import time