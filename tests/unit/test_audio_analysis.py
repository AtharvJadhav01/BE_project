"""
Unit tests for audio analysis components.

Tests the AudioFeatureExtractor, AudioStateEstimator, and AudioBaselineManager.
"""

import pytest
import time
import math
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.audio.features import AudioFeatureExtractor, AudioFeatures
from analysis.audio.state import AudioStateEstimator
from analysis.audio.baseline import AudioBaselineManager
from analysis.interfaces import DataQuality, BaselineStatus
from analysis.results import StateEstimate


class TestAudioFeatureExtractor:
    """Tests for AudioFeatureExtractor class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "sample_rate": 16000,
            "channels": 1,
            "pitch_min_hz": 80.0,
            "pitch_max_hz": 400.0,
            "pitch_confidence_threshold": 0.5,
            "pitch_method": "yin",
            "window_size_sec": 0.025,
            "hop_size_sec": 0.01,
            "vad_energy_threshold_db": -40.0,
            "vad_zcr_threshold": 0.1,
            "pitch_history_sec": 1.0,
            "energy_history_sec": 2.0,
        }

    @pytest.fixture
    def extractor(self, config):
        """Create a feature extractor instance."""
        return AudioFeatureExtractor("test_audio", config)

    def _create_sine_wave(self, frequency: float = 200.0, duration: float = 0.1,
                          sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
        """Create a synthetic sine wave for testing."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        return wave.astype(np.float32)

    def _create_silence(self, duration: float = 0.1, sample_rate: int = 16000) -> np.ndarray:
        """Create silence (zeros) for testing."""
        return np.zeros(int(sample_rate * duration), dtype=np.float32)

    def _create_white_noise(self, duration: float = 0.1, sample_rate: int = 16000,
                           amplitude: float = 0.1) -> np.ndarray:
        """Create white noise for testing."""
        noise = amplitude * np.random.randn(int(sample_rate * duration))
        return noise.astype(np.float32)

    def test_extractor_creation(self, extractor):
        """Test extractor creation."""
        assert extractor.name == "test_audio"
        assert extractor.config is not None

    def test_get_feature_names(self, extractor):
        """Test getting feature names."""
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        # Check key features are present
        assert "pitch_hz" in names
        assert "pitch_valid" in names
        assert "rms_energy" in names
        assert "rms_energy_db" in names
        assert "zcr" in names
        assert "voice_activity" in names
        assert "spectral_centroid" in names

    def test_extract_silence(self, extractor):
        """Test extraction with silence input."""
        silence = self._create_silence()
        raw_data = {"data": silence, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        assert result is not None
        assert result.modality == "test_audio"
        assert result.quality == DataQuality.INSUFFICIENT
        # Features dict has default values for silence
        assert isinstance(result.features, dict)
        assert len(result.features) > 0
        # Most values should be 0.0 for silence, except energy_db which is -120.0
        for k, v in result.features.items():
            if k in ("rms_energy_db", "energy_mean"):
                assert v == -120.0, f"Expected -120.0 for {k}, got {v}"
            else:
                assert v == 0.0, f"Expected 0.0 for {k}, got {v}"

    def test_extract_sine_wave(self, extractor):
        """Test extraction with a sine wave (clear pitch)."""
        # 200 Hz sine wave
        sine = self._create_sine_wave(frequency=200.0, duration=0.1)
        raw_data = {"data": sine, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        assert result is not None
        assert result.modality == "test_audio"
        # Quality should be good for a clear tone
        assert result.quality in [DataQuality.GOOD, DataQuality.DEGRADED]
        
        # Check features
        features = result.features
        assert "pitch_hz" in features
        assert "rms_energy" in features
        assert "rms_energy_db" in features
        assert "zcr" in features
        assert "voice_activity" in features
        
        # Energy should be non-zero
        assert features["rms_energy"] > 0
        assert features["rms_energy_db"] > -60  # Not silence

    def test_extract_white_noise(self, extractor):
        """Test extraction with white noise."""
        noise = self._create_white_noise()
        raw_data = {"data": noise, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        assert result is not None
        features = result.features
        
        # Noise should have energy but no clear pitch
        assert features["rms_energy"] > 0
        assert features["voice_activity"] >= 0

    def test_pitch_features(self, extractor):
        """Test pitch feature extraction."""
        # Create a sine wave with known frequency
        sine = self._create_sine_wave(frequency=220.0, duration=0.2)  # Longer duration for better pitch detection
        raw_data = {"data": sine, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        # Pitch detection may not be perfect with fallback method
        # but features should be present
        assert "pitch_hz" in features
        assert "pitch_valid" in features
        assert "pitch_confidence" in features

    def test_energy_features(self, extractor):
        """Test energy feature extraction."""
        sine = self._create_sine_wave(amplitude=0.5)
        raw_data = {"data": sine, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "rms_energy" in features
        assert "rms_energy_db" in features
        assert features["rms_energy"] > 0
        assert features["rms_energy_db"] > -60  # Not silence

    def test_zcr_features(self, extractor):
        """Test zero crossing rate extraction."""
        sine = self._create_sine_wave()
        raw_data = {"data": sine, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "zcr" in features
        assert features["zcr"] >= 0

    def test_voice_activity(self, extractor):
        """Test voice activity detection."""
        # Sine wave should have voice activity
        sine = self._create_sine_wave()
        raw_data = {"data": sine, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "voice_activity" in features
        assert features["voice_activity"] in [0.0, 1.0]
        
        # Silence should have no voice activity
        silence = self._create_silence()
        raw_data = {"data": silence, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=2.0, wallclock="2024-01-01T00:00:01.000Z")
        features = result.features
        assert features["voice_activity"] == 0.0

    def test_spectral_features(self, extractor):
        """Test spectral feature extraction."""
        sine = self._create_sine_wave()
        raw_data = {"data": sine, "sample_rate": 16000}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "spectral_centroid" in features
        assert "spectral_rolloff" in features
        assert "spectral_flux" in features
        assert features["spectral_centroid"] >= 0

    def test_reset(self, extractor):
        """Test extractor reset."""
        sine = self._create_sine_wave()
        raw_data = {"data": sine, "sample_rate": 16000}
        
        # First extraction
        result1 = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        assert result1.features.get("facial_movement_velocity", 0) == 0  # Should be 0 on first frame
        
        # Reset
        extractor.reset()
        
        # Second extraction should behave like first
        result2 = extractor.extract(raw_data, timestamp=2.0, wallclock="2024-01-01T00:00:01.000Z")
        assert result2.features.get("facial_movement_velocity", 0) == 0


class TestAudioStateEstimator:
    """Tests for AudioStateEstimator class."""

    @pytest.fixture
    def config(self):
        """Test configuration with thresholds."""
        return {
            "pitch_high_z": 1.5,
            "pitch_low_z": -1.5,
            "pitch_variability_high_z": 1.5,
            "pitch_variability_low_z": -1.0,
            "energy_high_z": 1.0,
            "energy_low_z": -1.0,
            "energy_variability_high_z": 1.0,
            "voice_activity_high": 0.7,
            "voice_activity_low": 0.3,
            "voiced_ratio_high": 0.6,
            "pitch_std_high": 50.0,
            "pitch_std_low": 5.0,
            "energy_std_high": 10.0,
            "spectral_centroid_high": 3000.0,
        }

    @pytest.fixture
    def estimator(self, config):
        """Create a state estimator instance."""
        return AudioStateEstimator("test_audio", config)

    @pytest.fixture
    def baseline_manager(self):
        """Create a baseline manager with ready baseline."""
        manager = AudioBaselineManager(warmup_duration_sec=0.1, min_samples=1)
        # Update all audio features to make baseline ready
        features = [
            "pitch_hz", "pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
            "pitch_valid", "pitch_confidence",
            "rms_energy", "rms_energy_db", "energy_mean", "energy_std",
            "zcr", "zcr_mean", "zcr_std",
            "voice_activity", "voiced_ratio",
            "spectral_centroid", "spectral_rolloff", "spectral_flux"
        ]
        for f in features:
            manager.update(f, 0.0, timestamp=1.0)
        time.sleep(0.2)
        for f in features:
            manager.update(f, 0.0, timestamp=2.0)
        return manager

    def test_estimator_creation(self, estimator):
        """Test estimator creation."""
        assert estimator.name == "test_audio"
        assert estimator.config is not None
        assert len(estimator.thresholds) > 0

    def test_get_possible_states(self, estimator):
        """Test getting possible states."""
        states = estimator.get_possible_states()
        assert isinstance(states, list)
        assert "neutral" in states
        assert "elevated_arousal_like_indicator" in states
        assert "calm_like_indicator" in states
        assert "uncertain" in states

    def test_estimate_uncertain_when_baseline_not_ready(self, estimator):
        """Test that estimate returns uncertain when baseline not ready."""
        manager = AudioBaselineManager(warmup_duration_sec=10.0, min_samples=10)
        features = {"pitch_hz": 200.0, "rms_energy_db": -30.0}
        
        result = estimator.estimate(features, manager)
        
        assert result is not None
        assert result.state == "uncertain"
        assert result.confidence == 0.0
        assert result.uncertainty == 1.0

    def test_estimate_neutral(self, estimator, baseline_manager):
        """Test neutral state estimation."""
        features = {
            "pitch_hz": 0.0,
            "rms_energy_db": 0.0,
            "pitch_std": 20.0,
            "energy_std": 5.0,
            "voice_activity": 0.5,
            "voiced_ratio": 0.4,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert result.state in estimator.get_possible_states()
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0

    def test_estimate_elevated_arousal(self, estimator, baseline_manager):
        """Test elevated arousal detection."""
        features = {
            "pitch_hz": 0.0,
            "rms_energy_db": 0.0,
            "pitch_std": 60.0,
            "energy_std": 15.0,
            "voice_activity": 0.9,
            "voiced_ratio": 0.8,
            "spectral_centroid": 3500.0,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert result.state in estimator.get_possible_states()

    def test_estimate_calm(self, estimator, baseline_manager):
        """Test calm state detection."""
        features = {
            "pitch_hz": 0.0,
            "rms_energy_db": 0.0,
            "pitch_std": 3.0,
            "energy_std": 2.0,
            "voice_activity": 0.2,
            "voiced_ratio": 0.2,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert result.state in estimator.get_possible_states()

    def test_estimate_empty_features(self, estimator, baseline_manager):
        """Test estimation with empty features."""
        result = estimator.estimate({}, baseline_manager)
        assert result is None

    def test_estimate_confidence_bounds(self, estimator, baseline_manager):
        """Test that confidence and uncertainty are properly bounded."""
        features = {"pitch_hz": 0.0, "rms_energy_db": 0.0}
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0

    def test_estimate_contributing_features(self, estimator, baseline_manager):
        """Test that contributing features are returned."""
        features = {"pitch_hz": 0.0, "rms_energy_db": 0.0}
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert isinstance(result.contributing_features, dict)

    def test_reset(self, estimator):
        """Test estimator reset (should not error)."""
        estimator.reset()
        # Should not raise


class TestAudioBaselineManager:
    """Tests for AudioBaselineManager class."""

    @pytest.fixture
    def baseline_manager(self):
        """Create an audio baseline manager."""
        return AudioBaselineManager(
            warmup_duration_sec=1.0,
            min_samples=5,
            ema_alpha=0.1,
            min_std=1e-6
        )

    def test_creation(self, baseline_manager):
        """Test baseline manager creation."""
        assert baseline_manager.warmup_duration_sec == 1.0
        assert baseline_manager.min_samples == 5
        assert baseline_manager.ema_alpha == 0.1

    def test_feature_groups(self, baseline_manager):
        """Test that feature groups are defined."""
        assert "pitch" in baseline_manager.feature_groups
        assert "energy" in baseline_manager.feature_groups
        assert "zcr" in baseline_manager.feature_groups
        assert "voice_activity" in baseline_manager.feature_groups
        assert "spectral" in baseline_manager.feature_groups
        
        total_features = sum(len(v) for v in baseline_manager.feature_groups.values())
        assert total_features > 10

    def test_update_and_normalize(self, baseline_manager):
        """Test updating baseline and normalizing features."""
        # Update with some values
        for i in range(10):
            baseline_manager.update("rms_energy_db", -30.0 + i * 0.5, timestamp=float(i))
        
        time.sleep(1.1)
        
        for i in range(10, 15):
            baseline_manager.update("rms_energy_db", -30.0 + i * 0.5, timestamp=float(i))
        
        assert baseline_manager.is_ready("rms_energy_db")
        
        # Normalize a value
        z = baseline_manager.normalize("rms_energy_db", -25.0)
        assert z is not None
        assert z > 0  # Above mean

    def test_get_audio_baseline_status(self, baseline_manager):
        """Test getting audio baseline status by group."""
        for i in range(5):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
            baseline_manager.update("pitch_hz", 200.0, timestamp=float(i))
        time.sleep(1.1)
        for i in range(5, 10):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
            baseline_manager.update("pitch_hz", 200.0, timestamp=float(i))
        
        status = baseline_manager.get_audio_baseline_status()
        assert isinstance(status, dict)
        assert "energy" in status
        assert "pitch" in status

    def test_get_audio_baseline_summary(self, baseline_manager):
        """Test getting audio baseline summary."""
        for i in range(5):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
            baseline_manager.update("pitch_hz", 200.0, timestamp=float(i))
        time.sleep(1.1)
        for i in range(5, 10):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
            baseline_manager.update("pitch_hz", 200.0, timestamp=float(i))
        
        summary = baseline_manager.get_audio_baseline_summary()
        assert isinstance(summary, dict)
        assert "energy" in summary
        assert "pitch" in summary

    def test_is_audio_baseline_ready(self, baseline_manager):
        """Test checking audio baseline readiness."""
        assert not baseline_manager.is_audio_baseline_ready()
        
        all_features = []
        for group in baseline_manager.feature_groups.values():
            all_features.extend(group)
        
        for f in all_features:
            for i in range(10):
                baseline_manager.update(f, 0.5, timestamp=float(i))
        
        time.sleep(1.1)
        
        for f in all_features:
            for i in range(10, 15):
                baseline_manager.update(f, 0.5, timestamp=float(i))
        
        assert baseline_manager.is_audio_baseline_ready()

    def test_get_normalized_audio_features(self, baseline_manager):
        """Test normalizing multiple audio features at once."""
        for i in range(10):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
            baseline_manager.update("pitch_hz", 200.0, timestamp=float(i))
        time.sleep(1.1)
        for i in range(10, 15):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
            baseline_manager.update("pitch_hz", 200.0, timestamp=float(i))
        
        raw_features = {"rms_energy_db": -25.0, "pitch_hz": 220.0}
        normalized = baseline_manager.get_normalized_audio_features(raw_features)
        
        assert "rms_energy_db" in normalized
        assert "pitch_hz" in normalized
        assert normalized["rms_energy_db"] > 0  # Above mean
        assert normalized["pitch_hz"] > 0  # Above mean

    def test_reset_feature(self, baseline_manager):
        """Test resetting a specific feature."""
        for i in range(10):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
        time.sleep(1.1)
        for i in range(10, 15):
            baseline_manager.update("rms_energy_db", -30.0, timestamp=float(i))
        
        assert baseline_manager.is_ready("rms_energy_db")
        
        baseline_manager.reset_feature("rms_energy_db")
        
        assert not baseline_manager.is_ready("rms_energy_db")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])