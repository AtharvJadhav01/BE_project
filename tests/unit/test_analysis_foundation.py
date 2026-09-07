"""
Unit tests for analysis foundation module.
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.temporal import TemporalWindow, MultiFeatureTemporalWindow, WindowStats
from analysis.baseline import BaseBaselineManager, FeatureBaseline, BaselineStatus
from analysis.results import (
    FeatureVector, StateEstimate, AnalysisResult, 
    MultimodalAnalysisResult, AnalysisConfig,
    DataQuality, BaselineStatus as ResultsBaselineStatus
)
from analysis.interfaces import DataQuality as InterfaceDataQuality


class TestTemporalWindow:
    """Tests for TemporalWindow class."""

    def test_creation(self):
        """Test window creation."""
        window = TemporalWindow(window_duration_sec=5.0)
        assert window.window_duration_sec == 5.0
        assert window.is_empty()

    def test_add_single_observation(self):
        """Test adding a single observation."""
        window = TemporalWindow(window_duration_sec=5.0)
        window.add(10.0, timestamp=1.0)
        
        stats = window.get_stats()
        assert stats.count == 1
        assert stats.mean == 10.0
        assert stats.min_value == 10.0
        assert stats.max_value == 10.0
        assert stats.latest == 10.0
        assert stats.latest_timestamp == 1.0

    def test_add_multiple_observations(self):
        """Test adding multiple observations."""
        window = TemporalWindow(window_duration_sec=5.0)
        window.add(10.0, timestamp=1.0)
        window.add(20.0, timestamp=2.0)
        window.add(30.0, timestamp=3.0)
        
        stats = window.get_stats()
        assert stats.count == 3
        assert stats.mean == 20.0
        assert stats.std > 0
        assert stats.min_value == 10.0
        assert stats.max_value == 30.0
        assert stats.latest == 30.0

    def test_expiration(self):
        """Test expiration of old observations."""
        window = TemporalWindow(window_duration_sec=2.0)
        window.add(10.0, timestamp=1.0)
        window.add(20.0, timestamp=2.0)
        window.add(30.0, timestamp=4.0)  # First observation should expire
        
        stats = window.get_stats()
        assert stats.count == 2  # Only 2nd and 3rd observations remain
        assert stats.mean == 25.0  # (20 + 30) / 2

    def test_empty_window_stats(self):
        """Test stats from empty window."""
        window = TemporalWindow(window_duration_sec=5.0)
        stats = window.get_stats()
        
        assert stats.count == 0
        assert not stats.is_valid()
        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_insufficient_observations_std(self):
        """Test std with single observation."""
        window = TemporalWindow(window_duration_sec=5.0)
        window.add(10.0, timestamp=1.0)
        
        stats = window.get_stats()
        assert stats.count == 1
        assert stats.std == 0.0  # No variance with single sample

    def test_max_observations_limit(self):
        """Test max observations limit."""
        window = TemporalWindow(window_duration_sec=10.0, max_observations=3)
        window.add(10.0, timestamp=1.0)
        window.add(20.0, timestamp=2.0)
        window.add(30.0, timestamp=3.0)
        window.add(40.0, timestamp=4.0)  # Should remove oldest
        
        stats = window.get_stats()
        assert stats.count == 3
        assert stats.mean == 30.0  # (20 + 30 + 40) / 3

    def test_clear(self):
        """Test clearing the window."""
        window = TemporalWindow(window_duration_sec=5.0)
        window.add(10.0, timestamp=1.0)
        window.add(20.0, timestamp=2.0)
        window.clear()
        
        assert window.is_empty()
        stats = window.get_stats()
        assert stats.count == 0

    def test_len(self):
        """Test len() returns observation count."""
        window = TemporalWindow(window_duration_sec=5.0)
        assert len(window) == 0
        window.add(10.0, timestamp=1.0)
        assert len(window) == 1
        window.add(20.0, timestamp=2.0)
        assert len(window) == 2

    def test_get_span(self):
        """Test getting time span of window."""
        window = TemporalWindow(window_duration_sec=5.0)
        window.add(10.0, timestamp=1.0)
        window.add(20.0, timestamp=3.0)
        
        assert window.get_span() == 2.0

    def test_get_values_and_timestamps(self):
        """Test getting values and timestamps."""
        window = TemporalWindow(window_duration_sec=5.0)
        window.add(10.0, timestamp=1.0)
        window.add(20.0, timestamp=2.0)
        
        values = window.get_values()
        timestamps = window.get_timestamps()
        
        assert values == [10.0, 20.0]
        assert timestamps == [1.0, 2.0]

    def test_auto_timestamp(self):
        """Test automatic timestamp when not provided."""
        window = TemporalWindow(window_duration_sec=5.0)
        before = time.monotonic()
        window.add(10.0)
        after = time.monotonic()
        
        stats = window.get_stats()
        assert before <= stats.latest_timestamp <= after


class TestMultiFeatureTemporalWindow:
    """Tests for MultiFeatureTemporalWindow class."""

    def test_creation(self):
        """Test multi-feature window creation."""
        window = MultiFeatureTemporalWindow(
            feature_names=["feature1", "feature2"],
            window_duration_sec=5.0
        )
        assert window.get_feature_names() == ["feature1", "feature2"]

    def test_add_multiple_features(self):
        """Test adding observations for multiple features."""
        window = MultiFeatureTemporalWindow(
            feature_names=["feature1", "feature2"],
            window_duration_sec=5.0
        )
        window.add({"feature1": 10.0, "feature2": 20.0}, timestamp=1.0)
        window.add({"feature1": 15.0, "feature2": 25.0}, timestamp=2.0)
        
        stats = window.get_stats()
        assert stats["feature1"].count == 2
        assert stats["feature1"].mean == 12.5
        assert stats["feature2"].count == 2
        assert stats["feature2"].mean == 22.5

    def test_ignore_unknown_features(self):
        """Test that unknown features are ignored."""
        window = MultiFeatureTemporalWindow(
            feature_names=["feature1"],
            window_duration_sec=5.0
        )
        window.add({"feature1": 10.0, "unknown": 999.0}, timestamp=1.0)
        
        stats = window.get_stats()
        assert stats["feature1"].count == 1
        assert "unknown" not in stats

    def test_get_all_stats_dict(self):
        """Test getting all stats as dictionaries."""
        window = MultiFeatureTemporalWindow(
            feature_names=["feature1"],
            window_duration_sec=5.0
        )
        window.add({"feature1": 10.0}, timestamp=1.0)
        
        all_stats = window.get_all_stats_dict()
        assert "feature1" in all_stats
        assert all_stats["feature1"]["count"] == 1


class TestWindowStats:
    """Tests for WindowStats dataclass."""

    def test_is_valid(self):
        """Test is_valid method."""
        stats = WindowStats(count=0)
        assert not stats.is_valid()
        
        stats = WindowStats(count=1)
        assert stats.is_valid()

    def test_to_dict(self):
        """Test to_dict conversion."""
        stats = WindowStats(
            count=3,
            mean=20.0,
            std=10.0,
            min_value=10.0,
            max_value=30.0,
            latest=30.0,
            latest_timestamp=3.0,
            oldest_timestamp=1.0
        )
        d = stats.to_dict()
        assert d["count"] == 3
        assert d["mean"] == 20.0
        assert d["std"] == 10.0


class TestBaseBaselineManager:
    """Tests for BaseBaselineManager class."""

    def test_creation(self):
        """Test baseline manager creation."""
        manager = BaseBaselineManager(
            warmup_duration_sec=10.0,
            min_samples=50,
            ema_alpha=0.1
        )
        assert manager.warmup_duration_sec == 10.0
        assert manager.min_samples == 50
        assert manager.ema_alpha == 0.1

    def test_update_first_observation(self):
        """Test updating with first observation."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=1)
        manager.update("pitch", 200.0, timestamp=1.0)
        
        baseline = manager.get_baseline("pitch")
        assert baseline is not None
        assert baseline["count"] == 1
        assert baseline["mean"] == 200.0
        assert baseline["std"] == 0.0

    def test_update_multiple_observations(self):
        """Test updating with multiple observations."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=3)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("pitch", 210.0, timestamp=2.0)
        manager.update("pitch", 220.0, timestamp=3.0)
        
        baseline = manager.get_baseline("pitch")
        assert baseline["count"] == 3
        assert baseline["mean"] == 210.0
        assert baseline["std"] > 0

    def test_warmup_status(self):
        """Test warmup status progression."""
        manager = BaseBaselineManager(warmup_duration_sec=10.0, min_samples=5)
        
        # First update - warming up
        manager.update("pitch", 200.0, timestamp=1.0)
        assert manager.get_status("pitch") == BaselineStatus.WARMING_UP
        
        # Not enough samples
        manager.update("pitch", 210.0, timestamp=2.0)
        manager.update("pitch", 220.0, timestamp=3.0)
        assert manager.get_status("pitch") == BaselineStatus.WARMING_UP
        
        # Enough samples but not enough time
        manager.update("pitch", 230.0, timestamp=4.0)
        manager.update("pitch", 240.0, timestamp=5.0)
        assert manager.get_status("pitch") == BaselineStatus.WARMING_UP
        
        # Enough time and samples
        manager.update("pitch", 250.0, timestamp=15.0)
        assert manager.get_status("pitch") == BaselineStatus.READY

    def test_normalize_not_ready(self):
        """Test normalize returns None when baseline not ready."""
        manager = BaseBaselineManager(warmup_duration_sec=10.0, min_samples=10)
        manager.update("pitch", 200.0, timestamp=1.0)
        
        result = manager.normalize("pitch", 210.0)
        assert result is None

    def test_normalize_ready(self):
        """Test normalize returns z-score when baseline ready."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=3)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("pitch", 210.0, timestamp=2.0)
        manager.update("pitch", 220.0, timestamp=3.0)
        
        # Mean = 210, sample std = 10 (for n=3: M2=200, var=200/2=100, std=10)
        z = manager.normalize("pitch", 220.0)
        assert z is not None
        assert abs(z - 1.0) < 0.01  # (220 - 210) / 10 = 1.0

    def test_zero_std_handling(self):
        """Test normalization with zero/near-zero std."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=3, min_std=0.001)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("pitch", 200.0, timestamp=2.0)
        manager.update("pitch", 200.0, timestamp=3.0)
        
        # All same values, std = 0, should use min_std
        z = manager.normalize("pitch", 200.0)
        assert z == 0.0  # (200 - 200) / min_std = 0

    def test_ema_behavior(self):
        """Test EMA updates."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=1, ema_alpha=0.5)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("pitch", 300.0, timestamp=2.0)
        
        ema = manager.get_ema("pitch")
        # EMA: first = 200, second = 0.5 * 300 + 0.5 * 200 = 250
        assert ema == 250.0

    def test_multiple_features(self):
        """Test tracking multiple features independently."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=2)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("energy", 0.5, timestamp=1.0)
        manager.update("pitch", 210.0, timestamp=2.0)
        manager.update("energy", 0.6, timestamp=2.0)
        
        assert manager.is_ready("pitch")
        assert manager.is_ready("energy")
        
        pitch_z = manager.normalize("pitch", 220.0)
        energy_z = manager.normalize("energy", 0.7)
        
        assert pitch_z is not None
        assert energy_z is not None

    def test_get_all_statuses(self):
        """Test getting all feature statuses."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=1)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("energy", 0.5, timestamp=2.0)  # After warmup duration
        # Update pitch again to refresh its status after warmup duration
        manager.update("pitch", 210.0, timestamp=2.0)
        
        statuses = manager.get_all_statuses()
        assert statuses["pitch"] == BaselineStatus.READY
        assert statuses["energy"] == BaselineStatus.READY

    def test_get_warmup_progress(self):
        """Test warmup progress calculation."""
        manager = BaseBaselineManager(warmup_duration_sec=10.0, min_samples=10)
        
        # Before any updates
        assert manager.get_warmup_progress(timestamp=0.0) == 0.0
        
        # After some updates but not enough time/samples
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("pitch", 210.0, timestamp=2.0)
        
        progress = manager.get_warmup_progress(timestamp=5.0)
        assert 0.0 < progress < 1.0

    def test_reset(self):
        """Test resetting all baselines."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=1)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.reset()
        
        assert manager.get_status("pitch") == BaselineStatus.NOT_STARTED
        assert manager.get_baseline("pitch") is None

    def test_reset_feature(self):
        """Test resetting a specific feature."""
        manager = BaseBaselineManager(warmup_duration_sec=1.0, min_samples=1)
        manager.update("pitch", 200.0, timestamp=1.0)
        manager.update("energy", 0.5, timestamp=2.0)  # After warmup duration
        manager.reset_feature("pitch")
        
        assert manager.get_status("pitch") == BaselineStatus.NOT_STARTED
        assert manager.get_status("energy") == BaselineStatus.READY

    def test_invalid_parameters(self):
        """Test validation of constructor parameters."""
        with pytest.raises(ValueError):
            BaseBaselineManager(warmup_duration_sec=0)
        with pytest.raises(ValueError):
            BaseBaselineManager(min_samples=0)
        with pytest.raises(ValueError):
            BaseBaselineManager(ema_alpha=0)
        with pytest.raises(ValueError):
            BaseBaselineManager(ema_alpha=1.5)
        with pytest.raises(ValueError):
            BaseBaselineManager(min_std=0)


class TestFeatureVector:
    """Tests for FeatureVector model."""

    def test_creation(self):
        """Test feature vector creation."""
        fv = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"pitch": 200.0, "energy": 0.5},
            modality="audio",
            quality=DataQuality.GOOD
        )
        assert fv.modality == "audio"
        assert fv.features["pitch"] == 200.0

    def test_to_dict(self):
        """Test serialization."""
        fv = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"pitch": 200.0},
            modality="audio",
            quality=DataQuality.GOOD
        )
        d = fv.to_dict()
        assert d["modality"] == "audio"
        assert d["features"]["pitch"] == 200.0
        assert d["quality"] == "good"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "timestamp": 1.0,
            "wallclock": "2024-01-01T00:00:00.000Z",
            "features": {"pitch": 200.0},
            "modality": "audio",
            "quality": "good",
            "source_info": {},
            "processing_time_ms": 0.0
        }
        fv = FeatureVector.from_dict(data)
        assert fv.modality == "audio"
        assert fv.features["pitch"] == 200.0


class TestStateEstimate:
    """Tests for StateEstimate model."""

    def test_creation(self):
        """Test state estimate creation."""
        se = StateEstimate(
            state="elevated",
            confidence=0.8,
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality="audio",
            contributing_features={"pitch_z": 1.5},
            uncertainty=0.2
        )
        assert se.state == "elevated"
        assert se.confidence == 0.8
        assert se.modality == "audio"

    def test_to_dict(self):
        """Test serialization."""
        se = StateEstimate(
            state="calm",
            confidence=0.9,
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality="audio"
        )
        d = se.to_dict()
        assert d["state"] == "calm"
        assert d["confidence"] == 0.9
        assert d["modality"] == "audio"
        assert d["baseline_status"] == "not_started"


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_creation(self):
        """Test analysis result creation."""
        fv = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"pitch": 200.0},
            modality="audio"
        )
        result = AnalysisResult(
            modality="audio",
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            feature_vector=fv
        )
        assert result.modality == "audio"
        assert result.get_timestamp() == 1.0
        assert result.get_modality() == "audio"

    def test_with_state_estimate(self):
        """Test analysis result with state estimate."""
        fv = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"pitch": 200.0},
            modality="audio"
        )
        se = StateEstimate(
            state="elevated",
            confidence=0.8,
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality="audio"
        )
        result = AnalysisResult(
            modality="audio",
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            feature_vector=fv,
            state_estimate=se
        )
        assert result.state_estimate is not None
        assert result.state_estimate.state == "elevated"

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        fv = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"pitch": 200.0},
            modality="audio"
        )
        se = StateEstimate(
            state="elevated",
            confidence=0.8,
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality="audio"
        )
        result = AnalysisResult(
            modality="audio",
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            feature_vector=fv,
            state_estimate=se
        )
        
        d = result.to_dict()
        restored = AnalysisResult.from_dict(d)
        
        assert restored.modality == "audio"
        assert restored.feature_vector.features["pitch"] == 200.0
        assert restored.state_estimate.state == "elevated"
        assert restored.state_estimate.confidence == 0.8


class TestMultimodalAnalysisResult:
    """Tests for MultimodalAnalysisResult model."""

    def test_creation(self):
        """Test multimodal result creation."""
        fv_audio = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"pitch": 200.0},
            modality="audio"
        )
        fv_face = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={"smile": 0.8},
            modality="face"
        )
        
        result = MultimodalAnalysisResult(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality_results={
                "audio": AnalysisResult(
                    modality="audio",
                    timestamp=1.0,
                    wallclock="2024-01-01T00:00:00.000Z",
                    feature_vector=fv_audio
                ),
                "face": AnalysisResult(
                    modality="face",
                    timestamp=1.0,
                    wallclock="2024-01-01T00:00:00.000Z",
                    feature_vector=fv_face
                )
            }
        )
        assert "audio" in result.modality_results
        assert "face" in result.modality_results

    def test_with_fused_state(self):
        """Test multimodal result with fused state."""
        fv = FeatureVector(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            features={},
            modality="fusion"
        )
        se = StateEstimate(
            state="stressed",
            confidence=0.85,
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality="fusion"
        )
        
        result = MultimodalAnalysisResult(
            timestamp=1.0,
            wallclock="2024-01-01T00:00:00.000Z",
            modality_results={},
            fused_state=se
        )
        assert result.fused_state is not None
        assert result.fused_state.state == "stressed"


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_defaults(self):
        """Test default configuration values."""
        config = AnalysisConfig()
        assert config.temporal_window_sec == 3.0
        assert config.update_interval_ms == 500
        assert config.baseline_warmup_sec == 10.0
        assert config.baseline_min_samples == 50
        assert config.ema_alpha == 0.1

    def test_serialization(self):
        """Test config serialization."""
        config = AnalysisConfig(temporal_window_sec=5.0)
        d = config.to_dict()
        assert d["temporal_window_sec"] == 5.0

    def test_from_dict(self):
        """Test config deserialization."""
        data = {
            "temporal_window_sec": 4.0,
            "update_interval_ms": 1000,
            "baseline_warmup_sec": 15.0,
            "baseline_min_samples": 100,
            "ema_alpha": 0.2,
            "confidence_threshold": 0.7,
            "uncertainty_threshold": 0.3,
            "min_modalities_for_fusion": 3
        }
        config = AnalysisConfig.from_dict(data)
        assert config.temporal_window_sec == 4.0
        assert config.update_interval_ms == 1000


class TestDataQualityEnum:
    """Tests for DataQuality enum."""

    def test_values(self):
        """Test enum values."""
        assert DataQuality.GOOD == "good"
        assert DataQuality.DEGRADED == "degraded"
        assert DataQuality.POOR == "poor"
        assert DataQuality.INSUFFICIENT == "insufficient"

    def test_interface_compatibility(self):
        """Test compatibility with interface enum."""
        assert InterfaceDataQuality.GOOD == DataQuality.GOOD


class TestBaselineStatusEnum:
    """Tests for BaselineStatus enum."""

    def test_values(self):
        """Test enum values."""
        assert BaselineStatus.NOT_STARTED == "not_started"
        assert BaselineStatus.WARMING_UP == "warming_up"
        assert BaselineStatus.READY == "ready"
        assert BaselineStatus.ERROR == "error"


# Import datetime for FeatureVector tests
from datetime import datetime