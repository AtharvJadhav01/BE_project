"""
Unit tests for face analysis components.

Tests the FaceFeatureExtractor, FaceStateEstimator, and FaceBaselineManager.
"""

import pytest
import time
import math
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.face.features import FaceFeatureExtractor, FacialFeatures
from analysis.face.state import FaceStateEstimator
from analysis.face.baseline import FaceBaselineManager
from analysis.interfaces import DataQuality, BaselineStatus
from analysis.results import StateEstimate


class TestFaceFeatureExtractor:
    """Tests for FaceFeatureExtractor class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "eye_aspect_ratio_threshold": 0.2,
            "blink_rate_window_sec": 10.0,
            "facial_movement_history_size": 10,
        }

    @pytest.fixture
    def extractor(self, config):
        """Create a feature extractor instance."""
        return FaceFeatureExtractor("test_face", config)

    def _create_mock_detection_result(self, face_detected=True):
        """Create a mock detection result with realistic data."""
        if not face_detected:
            return {
                "face_detected": False,
                "landmarks_3d": [],
                "blendshapes": {},
                "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "iris": None
            }
        
        # Create 468 landmarks with some variation
        landmarks_3d = []
        for i in range(468):
            # Add some realistic variation
            x = 0.5 + (np.random.random() - 0.5) * 0.1
            y = 0.5 + (np.random.random() - 0.5) * 0.1
            z = (np.random.random() - 0.5) * 0.05
            landmarks_3d.append({"x": x, "y": y, "z": z})
        
        # Ensure key landmarks have specific values for testing
        # Left eye outer corner (33)
        landmarks_3d[33] = {"x": 0.35, "y": 0.45, "z": 0.0}
        # Right eye outer corner (263)
        landmarks_3d[263] = {"x": 0.65, "y": 0.45, "z": 0.0}
        # Left eye inner (133)
        landmarks_3d[133] = {"x": 0.42, "y": 0.45, "z": 0.0}
        # Right eye inner (362)
        landmarks_3d[362] = {"x": 0.58, "y": 0.45, "z": 0.0}
        # Nose tip (1)
        landmarks_3d[1] = {"x": 0.5, "y": 0.55, "z": 0.02}
        # Chin (152)
        landmarks_3d[152] = {"x": 0.5, "y": 0.75, "z": 0.0}
        # Mouth corners (61, 291)
        landmarks_3d[61] = {"x": 0.42, "y": 0.65, "z": 0.0}
        landmarks_3d[291] = {"x": 0.58, "y": 0.65, "z": 0.0}
        
        # Blendshapes with all 52 values
        blendshapes = {
            "browDownLeft": 0.0, "browDownRight": 0.0, "browInnerUp": 0.0,
            "browOuterUpLeft": 0.0, "browOuterUpRight": 0.0,
            "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0,
            "eyeLookDownLeft": 0.0, "eyeLookDownRight": 0.0,
            "eyeLookInLeft": 0.0, "eyeLookInRight": 0.0,
            "eyeLookOutLeft": 0.0, "eyeLookOutRight": 0.0,
            "eyeLookUpLeft": 0.0, "eyeLookUpRight": 0.0,
            "eyeSquintLeft": 0.0, "eyeSquintRight": 0.0,
            "eyeWideLeft": 0.0, "eyeWideRight": 0.0,
            "jawForward": 0.0, "jawLeft": 0.0, "jawOpen": 0.0,
            "jawRight": 0.0, "mouthClose": 0.0,
            "mouthDimpleLeft": 0.0, "mouthDimpleRight": 0.0,
            "mouthFrownLeft": 0.0, "mouthFrownRight": 0.0,
            "mouthFunnel": 0.0, "mouthLeft": 0.0,
            "mouthLowerDownLeft": 0.0, "mouthLowerDownRight": 0.0,
            "mouthPressLeft": 0.0, "mouthPressRight": 0.0,
            "mouthPucker": 0.0, "mouthRight": 0.0,
            "mouthRollLower": 0.0, "mouthRollUpper": 0.0,
            "mouthShrugLower": 0.0, "mouthShrugUpper": 0.0,
            "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0,
            "mouthStretchLeft": 0.0, "mouthStretchRight": 0.0,
            "mouthUpperUpLeft": 0.0, "mouthUpperUpRight": 0.0,
            "noseSneerLeft": 0.0, "noseSneerRight": 0.0
        }
        
        return {
            "face_detected": True,
            "landmarks_3d": landmarks_3d,
            "blendshapes": blendshapes,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "iris": {"left": {"x": 0.4, "y": 0.3, "z": 0.0}, "right": {"x": 0.6, "y": 0.3, "z": 0.0}}
        }

    def test_extractor_creation(self, extractor):
        """Test extractor creation."""
        assert extractor.name == "test_face"
        assert extractor.config is not None

    def test_get_feature_names(self, extractor):
        """Test getting feature names."""
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        # Check key features are present
        assert "eye_openness_left" in names
        assert "eye_openness_right" in names
        assert "mouth_openness" in names
        assert "mouth_width" in names
        assert "brow_inner_up" in names
        assert "head_pitch" in names
        assert "head_yaw" in names
        assert "head_roll" in names
        assert "facial_movement_velocity" in names
        assert "facial_symmetry" in names
        assert "overall_expressivity" in names
        # New emotion estimation features
        assert "mouth_funnel" in names
        assert "mouth_pucker" in names
        assert "mouth_stretch_left" in names
        assert "mouth_stretch_right" in names
        assert "nose_sneer_left" in names
        assert "nose_sneer_right" in names

    def test_extract_no_face(self, extractor):
        """Test extraction when no face is detected."""
        raw_data = {"face_detected": False, "landmarks_3d": [], "blendshapes": {}, "head_pose": {}}
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        assert result is not None
        assert result.modality == "test_face"
        assert result.quality == DataQuality.INSUFFICIENT
        # Features dict has default zero values (except facial_symmetry=1.0)
        assert isinstance(result.features, dict)
        assert len(result.features) > 0
        # Most values should be 0.0 for no-face case, facial_symmetry=1.0
        for k, v in result.features.items():
            if k == "facial_symmetry":
                assert v == 1.0
            else:
                assert v == 0.0

    def test_extract_with_face(self, extractor):
        """Test extraction with face detected."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        assert result is not None
        assert result.modality == "test_face"
        assert result.quality in [DataQuality.GOOD, DataQuality.DEGRADED]
        assert isinstance(result.features, dict)
        assert len(result.features) > 0
        
        # Check key features exist
        assert "eye_openness_left" in result.features
        assert "eye_openness_right" in result.features
        assert "mouth_openness" in result.features
        assert "mouth_width" in result.features
        assert "head_pitch" in result.features
        assert "facial_symmetry" in result.features

    def test_eye_features(self, extractor):
        """Test eye feature extraction."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        # Eye openness should be in valid range
        assert 0.0 <= features.get("eye_openness_left", 0) <= 1.0
        assert 0.0 <= features.get("eye_openness_right", 0) <= 1.0
        assert 0.0 <= features.get("eye_openness_avg", 0) <= 1.0
        # EAR should be positive
        assert features.get("eye_aspect_ratio_left", 0) >= 0
        assert features.get("eye_aspect_ratio_right", 0) >= 0

    def test_mouth_features(self, extractor):
        """Test mouth feature extraction."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "mouth_openness" in features
        assert "mouth_width" in features
        assert "lip_press" in features
        assert "smile_asymmetry" in features
        # New mouth features for emotion estimation
        assert "mouth_funnel" in features
        assert "mouth_pucker" in features
        assert "mouth_stretch_left" in features
        assert "mouth_stretch_right" in features
        # Values should be non-negative
        assert features.get("mouth_openness", 0) >= 0
        assert features.get("lip_press", 0) >= 0
        assert features.get("mouth_funnel", 0) >= 0
        assert features.get("mouth_pucker", 0) >= 0
        assert features.get("mouth_stretch_left", 0) >= 0
        assert features.get("mouth_stretch_right", 0) >= 0

    def test_eyebrow_features(self, extractor):
        """Test eyebrow feature extraction."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "brow_inner_up" in features
        assert "brow_down_left" in features
        assert "brow_down_right" in features

    def test_nose_features(self, extractor):
        """Test nose feature extraction for emotion estimation."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "nose_sneer_left" in features
        assert "nose_sneer_right" in features
        # Values should be non-negative
        assert features.get("nose_sneer_left", 0) >= 0
        assert features.get("nose_sneer_right", 0) >= 0

    def test_head_pose_features(self, extractor):
        """Test head pose feature extraction."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "head_pitch" in features
        assert "head_yaw" in features
        assert "head_roll" in features
        # Should be in reasonable degree range
        assert -90 <= features.get("head_pitch", 0) <= 90
        assert -90 <= features.get("head_yaw", 0) <= 90
        assert -90 <= features.get("head_roll", 0) <= 90

    def test_facial_movement(self, extractor):
        """Test facial movement velocity calculation."""
        raw_data = self._create_mock_detection_result()
        
        # First extraction - no previous frame
        result1 = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        # Movement should be 0 on first frame
        assert result1.features.get("facial_movement_velocity", 0) == 0.0
        
        # Second extraction - small time delta
        raw_data2 = self._create_mock_detection_result()
        result2 = extractor.extract(raw_data2, timestamp=1.033, wallclock="2024-01-01T00:00:00.033Z")
        # Should have some movement calculation (may be 0 if landmarks identical)
        assert "facial_movement_velocity" in result2.features

    def test_facial_symmetry(self, extractor):
        """Test facial symmetry calculation."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "facial_symmetry" in features
        # Symmetry should be in 0-1 range
        assert 0.0 <= features.get("facial_symmetry", 0) <= 1.0

    def test_overall_expressivity(self, extractor):
        """Test overall expressivity calculation."""
        raw_data = self._create_mock_detection_result()
        
        result = extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        features = result.features
        assert "overall_expressivity" in features
        assert 0.0 <= features.get("overall_expressivity", 0) <= 1.0

    def test_reset(self, extractor):
        """Test extractor reset."""
        raw_data = self._create_mock_detection_result()
        extractor.extract(raw_data, timestamp=1.0, wallclock="2024-01-01T00:00:00.000Z")
        
        extractor.reset()
        
        # After reset, should behave like first extraction
        result = extractor.extract(raw_data, timestamp=2.0, wallclock="2024-01-01T00:00:01.000Z")
        assert result.features.get("facial_movement_velocity", 0) == 0.0


class TestFaceStateEstimator:
    """Tests for FaceStateEstimator class."""

    @pytest.fixture
    def config(self):
        """Test configuration with thresholds."""
        return {
            "eye_openness_high": 0.7,
            "eye_openness_low": 0.3,
            "blink_rate_high": 25.0,
            "blink_rate_low": 8.0,
            "mouth_openness_high": 0.6,
            "mouth_width_high": 0.4,
            "mouth_width_low": -0.2,
            "lip_press_high": 0.5,
            "brow_inner_up_high": 0.4,
            "brow_down_high": 0.4,
            "brow_outer_up_high": 0.3,
            "facial_movement_high": 0.02,
            "facial_symmetry_low": 0.7,
            "expressivity_high": 0.6,
            "head_pitch_high": 15.0,
            "head_pitch_low": -15.0,
            "head_yaw_high": 20.0,
            "head_yaw_low": -20.0,
        }

    @pytest.fixture
    def estimator(self, config):
        """Create a state estimator instance."""
        return FaceStateEstimator("test_face", config)

    @pytest.fixture
    def baseline_manager(self):
        """Create a baseline manager with ready baseline."""
        manager = FaceBaselineManager(warmup_duration_sec=0.1, min_samples=1)
        # Update all face features to make baseline ready
        features = [
            "eye_openness_left", "eye_openness_right", "eye_openness_avg",
            "eye_aspect_ratio_left", "eye_aspect_ratio_right", "blink_rate",
            "mouth_openness", "mouth_width", "lip_press", "smile_asymmetry",
            "brow_inner_up", "brow_down_left", "brow_down_right",
            "brow_outer_up_left", "brow_outer_up_right",
            "head_pitch", "head_yaw", "head_roll",
            "facial_movement_velocity", "facial_symmetry",
            "overall_expressivity"
        ]
        for f in features:
            manager.update(f, 0.0, timestamp=1.0)
        # Wait for warmup
        time.sleep(0.2)
        for f in features:
            manager.update(f, 0.0, timestamp=2.0)
        return manager

    def test_estimator_creation(self, estimator):
        """Test estimator creation."""
        assert estimator.name == "test_face"
        assert estimator.config is not None
        assert len(estimator.thresholds) > 0

    def test_get_possible_states(self, estimator):
        """Test getting possible states."""
        states = estimator.get_possible_states()
        assert isinstance(states, list)
        assert "neutral" in states
        assert "positive_expression_indicator" in states
        assert "negative_expression_indicator" in states
        assert "surprise_like_indicator" in states
        assert "elevated_facial_tension_indicator" in states
        assert "uncertain" in states

    def test_estimate_uncertain_when_baseline_not_ready(self, estimator):
        """Test that estimate returns uncertain when baseline not ready."""
        manager = FaceBaselineManager(warmup_duration_sec=10.0, min_samples=10)
        features = {"eye_openness_avg": 0.5, "mouth_width": 0.0}
        
        result = estimator.estimate(features, manager)
        
        assert result is not None
        assert result.state == "uncertain"
        assert result.confidence == 0.0
        assert result.uncertainty == 1.0

    def test_estimate_neutral(self, estimator, baseline_manager):
        """Test neutral state estimation."""
        # Neutral features: moderate eye openness, relaxed mouth, symmetric
        features = {
            "eye_openness_avg": 0.5,
            "mouth_width": 0.0,
            "brow_inner_up": 0.1,
            "brow_down_left": 0.1,
            "brow_down_right": 0.1,
            "facial_symmetry": 0.9,
            "facial_movement_velocity": 0.005,
            "lip_press": 0.1,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        # Should be one of the known states
        assert result.state in estimator.get_possible_states()
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0

    def test_estimate_positive_expression(self, estimator, baseline_manager):
        """Test positive expression (smile) detection."""
        features = {
            "mouth_width": 0.6,  # Strong smile
            "eye_openness_avg": 0.5,
            "smile_asymmetry": 0.1,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        # Could be positive_expression_indicator or neutral depending on other features
        assert result.state in estimator.get_possible_states()

    def test_estimate_surprise_like(self, estimator, baseline_manager):
        """Test surprise-like detection."""
        features = {
            "brow_inner_up": 0.6,  # High inner brow
            "eye_openness_avg": 0.8,  # Wide eyes
            "mouth_openness": 0.7,  # Open mouth
            "facial_symmetry": 0.9,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        # Should detect surprise if thresholds met
        assert result.state in estimator.get_possible_states()

    def test_estimate_negative_expression(self, estimator, baseline_manager):
        """Test negative expression (frown) detection."""
        features = {
            "brow_down_left": 0.6,
            "brow_down_right": 0.6,
            "lip_press": 0.6,
            "brow_inner_up": 0.1,  # Not surprised
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert result.state in estimator.get_possible_states()

    def test_estimate_elevated_tension(self, estimator, baseline_manager):
        """Test elevated facial tension detection."""
        features = {
            "facial_movement_velocity": 0.05,  # High movement
            "facial_symmetry": 0.5,  # Low symmetry
            "lip_press": 0.6,
            "brow_down_left": 0.5,
            "brow_down_right": 0.5,
            "mouth_openness": 0.3,  # Not extreme surprise
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
        features = {"eye_openness_avg": 0.5}
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0

    def test_estimate_contributing_features(self, estimator, baseline_manager):
        """Test that contributing features are returned."""
        features = {
            "mouth_width": 0.5,
            "eye_openness_avg": 0.5,
        }
        
        result = estimator.estimate(features, baseline_manager)
        
        assert result is not None
        assert isinstance(result.contributing_features, dict)

    def test_reset(self, estimator):
        """Test estimator reset (should not error)."""
        estimator.reset()
        # Should not raise


class TestFaceBaselineManager:
    """Tests for FaceBaselineManager class."""

    @pytest.fixture
    def baseline_manager(self):
        """Create a face baseline manager."""
        return FaceBaselineManager(
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
        assert "eye" in baseline_manager.feature_groups
        assert "mouth" in baseline_manager.feature_groups
        assert "eyebrow" in baseline_manager.feature_groups
        assert "head_pose" in baseline_manager.feature_groups
        assert "movement" in baseline_manager.feature_groups
        assert "expressivity" in baseline_manager.feature_groups
        
        # Check total features
        total_features = sum(len(v) for v in baseline_manager.feature_groups.values())
        assert total_features > 10

    def test_update_and_normalize(self, baseline_manager):
        """Test updating baseline and normalizing features."""
        # Update with some values
        for i in range(10):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        
        # Wait for warmup
        time.sleep(1.1)
        
        # Update more to reach min_samples
        for i in range(10, 15):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        
        # Should be ready
        assert baseline_manager.is_ready("eye_openness_avg")
        
        # Normalize a value
        z = baseline_manager.normalize("eye_openness_avg", 0.6)
        assert z is not None
        # 0.6 vs mean 0.5, should be positive z-score
        assert z > 0

    def test_get_face_baseline_status(self, baseline_manager):
        """Test getting face baseline status by group."""
        for i in range(5):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        time.sleep(1.1)
        for i in range(5, 10):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        
        status = baseline_manager.get_face_baseline_status()
        assert isinstance(status, dict)
        assert "eye" in status
        assert "mouth" in status

    def test_get_face_baseline_summary(self, baseline_manager):
        """Test getting face baseline summary."""
        for i in range(5):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        time.sleep(1.1)
        for i in range(5, 10):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        
        summary = baseline_manager.get_face_baseline_summary()
        assert isinstance(summary, dict)
        assert "eye" in summary

    def test_is_face_baseline_ready(self, baseline_manager):
        """Test checking face baseline readiness."""
        # Not ready initially
        assert not baseline_manager.is_face_baseline_ready()
        
        # Update all features
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
        
        # Should be ready
        assert baseline_manager.is_face_baseline_ready()

    def test_get_normalized_face_features(self, baseline_manager):
        """Test normalizing multiple face features at once."""
        for i in range(10):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
            baseline_manager.update("mouth_width", 0.0, timestamp=float(i))
        time.sleep(1.1)
        for i in range(10, 15):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
            baseline_manager.update("mouth_width", 0.0, timestamp=float(i))
        
        raw_features = {"eye_openness_avg": 0.6, "mouth_width": 0.1}
        normalized = baseline_manager.get_normalized_face_features(raw_features)
        
        assert "eye_openness_avg" in normalized
        assert "mouth_width" in normalized
        assert normalized["eye_openness_avg"] > 0  # Above mean
        assert normalized["mouth_width"] > 0  # Above mean

    def test_reset_feature(self, baseline_manager):
        """Test resetting a specific feature."""
        for i in range(10):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        time.sleep(1.1)
        for i in range(10, 15):
            baseline_manager.update("eye_openness_avg", 0.5, timestamp=float(i))
        
        assert baseline_manager.is_ready("eye_openness_avg")
        
        baseline_manager.reset_feature("eye_openness_avg")
        
        assert not baseline_manager.is_ready("eye_openness_avg")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])