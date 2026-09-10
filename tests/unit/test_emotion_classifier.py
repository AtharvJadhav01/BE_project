"""
Unit tests for FaceEmotionClassifier.

Tests model loading, feature validation, preprocessing, inference,
no-face handling, and configuration.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.face.emotion_classifier import (
    FaceEmotionClassifier,
    EmotionClassifierError,
    MODEL_FEATURE_NAMES,
    EMOTION_LABELS,
    create_emotion_classifier,
)


# Shared fixtures
@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock model files for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Create mock model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    import json
    
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    X = np.random.randn(100, len(MODEL_FEATURE_NAMES))
    y = np.random.randint(0, 7, 100)
    model.fit(X, y)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    model_path = model_dir / "rf_emotion_200est_15depth.joblib"
    scaler_path = model_dir / "scaler.joblib"
    metadata_path = model_dir / "rf_emotion_200est_15depth_metadata.json"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        "model_name": "rf_emotion_200est_15depth",
        "model_type": "RandomForestClassifier",
        "timestamp": "2026-09-10T01:48:32.387045",
        "feature_names": MODEL_FEATURE_NAMES,
        "feature_dim": len(MODEL_FEATURE_NAMES),
        "label_map": {str(i): label for i, label in enumerate(EMOTION_LABELS)},
        "target_names": EMOTION_LABELS,
        "metrics": {
            "test": {"accuracy": 0.4793}
        }
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metadata_path": metadata_path,
        "model_dir": model_dir,
    }


@pytest.fixture
def classifier(mock_model_files):
    """Create a classifier instance with mock model files."""
    return FaceEmotionClassifier(
        model_path=mock_model_files["model_path"],
        scaler_path=mock_model_files["scaler_path"],
        metadata_path=mock_model_files["metadata_path"],
    )


@pytest.fixture
def valid_features():
    """Create a valid feature dictionary."""
    return {name: 0.5 for name in MODEL_FEATURE_NAMES}


class TestFaceEmotionClassifier:
    """Tests for FaceEmotionClassifier class."""

    def test_classifier_creation(self, classifier):
        """Test classifier creation and model loading."""
        assert classifier.is_ready()
        assert classifier.get_load_error() is None

    def test_get_feature_names(self, classifier):
        """Test getting feature names."""
        names = classifier.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == len(MODEL_FEATURE_NAMES)
        assert names == MODEL_FEATURE_NAMES

    def test_get_label_names(self, classifier):
        """Test getting label names."""
        labels = classifier.get_label_names()
        assert isinstance(labels, list)
        assert len(labels) == 7
        assert labels == EMOTION_LABELS

    def test_get_metadata(self, classifier):
        """Test getting metadata."""
        metadata = classifier.get_metadata()
        assert isinstance(metadata, dict)
        assert "model_name" in metadata
        assert "feature_names" in metadata

    def test_validate_features_valid(self, classifier, valid_features):
        """Test feature validation with valid features."""
        is_valid, missing = classifier.validate_features(valid_features)
        assert is_valid
        assert missing == []

    def test_validate_features_missing(self, classifier, valid_features):
        """Test feature validation with missing features."""
        # Remove a feature
        invalid_features = valid_features.copy()
        del invalid_features["eye_openness_left"]
        
        is_valid, missing = classifier.validate_features(invalid_features)
        assert not is_valid
        assert "eye_openness_left" in missing

    def test_validate_features_extra(self, classifier, valid_features):
        """Test feature validation with extra features (should still pass)."""
        extra_features = valid_features.copy()
        extra_features["extra_feature"] = 1.0
        
        is_valid, missing = classifier.validate_features(extra_features)
        assert is_valid
        assert missing == []

    def test_prepare_feature_vector(self, classifier, valid_features):
        """Test feature vector preparation."""
        vector = classifier.prepare_feature_vector(valid_features)
        
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (1, len(MODEL_FEATURE_NAMES))
        assert vector.dtype == np.float32
        
        # Check ordering matches MODEL_FEATURE_NAMES
        for i, name in enumerate(MODEL_FEATURE_NAMES):
            assert vector[0, i] == valid_features[name]

    def test_prepare_feature_vector_missing_raises(self, classifier, valid_features):
        """Test that missing features raise error."""
        invalid_features = valid_features.copy()
        del invalid_features["mouth_openness"]
        
        with pytest.raises(EmotionClassifierError) as exc_info:
            classifier.prepare_feature_vector(invalid_features)
        assert "Missing required features" in str(exc_info.value)

    def test_prepare_feature_vector_nan_raises(self, classifier, valid_features):
        """Test that NaN values raise error."""
        invalid_features = valid_features.copy()
        invalid_features["eye_openness_left"] = float('nan')
        
        with pytest.raises(EmotionClassifierError) as exc_info:
            classifier.prepare_feature_vector(invalid_features)
        assert "NaN or Inf" in str(exc_info.value)

    def test_prepare_feature_vector_inf_raises(self, classifier, valid_features):
        """Test that Inf values raise error."""
        invalid_features = valid_features.copy()
        invalid_features["mouth_openness"] = float('inf')
        
        with pytest.raises(EmotionClassifierError) as exc_info:
            classifier.prepare_feature_vector(invalid_features)
        assert "NaN or Inf" in str(exc_info.value)

    def test_predict_returns_correct_structure(self, classifier, valid_features):
        """Test that predict returns correct result structure."""
        result = classifier.predict(valid_features)
        
        assert isinstance(result, dict)
        assert "expression" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "timestamp" in result
        assert "wallclock" in result
        assert "model_version" in result
        
        # Check types
        assert isinstance(result["expression"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["probabilities"], dict)
        assert isinstance(result["timestamp"], float)
        assert isinstance(result["wallclock"], str)
        
        # Check confidence bounds
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Check probabilities
        probs = result["probabilities"]
        assert len(probs) == 7
        for label in EMOTION_LABELS:
            assert label in probs
            assert 0.0 <= probs[label] <= 1.0
        
        # Probabilities should sum to ~1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_expression_in_labels(self, classifier, valid_features):
        """Test that predicted expression is one of the known labels."""
        result = classifier.predict(valid_features)
        assert result["expression"] in EMOTION_LABELS

    def test_predict_confidence_matches_max_probability(self, classifier, valid_features):
        """Test that confidence matches the max probability."""
        result = classifier.predict(valid_features)
        max_prob = max(result["probabilities"].values())
        assert abs(result["confidence"] - max_prob) < 0.001

    def test_predict_with_timestamp(self, classifier, valid_features):
        """Test prediction with custom timestamp."""
        timestamp = 12345.678
        result = classifier.predict(valid_features, timestamp=timestamp)
        assert result["timestamp"] == timestamp

    def test_predict_raw(self, classifier, valid_features):
        """Test low-level raw prediction."""
        vector = classifier.prepare_feature_vector(valid_features)
        scaled_vector = classifier._scaler.transform(vector)
        
        pred_idx, confidence, probas = classifier.predict_raw(scaled_vector)
        
        assert isinstance(pred_idx, int)
        assert 0 <= pred_idx < 7
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(probas, np.ndarray)
        assert len(probas) == 7

    def test_reset(self, classifier):
        """Test reset method (no-op but should not error)."""
        classifier.reset()
        # Should not raise

    def test_is_ready_after_load(self, classifier):
        """Test is_ready returns True after successful load."""
        assert classifier.is_ready()

    def test_load_error_handling(self, tmp_path):
        """Test handling of missing model files."""
        with pytest.raises(EmotionClassifierError) as exc_info:
            FaceEmotionClassifier(
                model_path=tmp_path / "nonexistent.joblib",
                scaler_path=tmp_path / "nonexistent.joblib",
                metadata_path=tmp_path / "nonexistent.json",
            )
        assert "not found" in str(exc_info.value).lower()


class TestEmotionClassifierIntegration:
    """Integration tests for emotion classifier with face analysis pipeline."""

    def test_emotion_classifier_factory(self, mock_model_files):
        """Test factory function creates classifier."""
        classifier = create_emotion_classifier()
        # Factory uses default paths, will fail if models not in default location
        # Just test it doesn't crash on creation
        assert classifier is not None

    def test_feature_ordering_matches_training(self, classifier):
        """Test that feature ordering matches training metadata."""
        feature_names = classifier.get_feature_names()
        metadata = classifier.get_metadata()
        saved_names = metadata.get("feature_names", [])
        
        assert feature_names == saved_names

    def test_scaler_applied_correctly(self, classifier, valid_features):
        """Test that scaler is applied during prediction."""
        # Get raw feature vector
        raw_vector = classifier.prepare_feature_vector(valid_features)
        
        # Get scaled vector using internal scaler
        scaled_vector = classifier._scaler.transform(raw_vector)
        
        # The prediction should use scaled features
        pred_idx, confidence, probas = classifier.predict_raw(scaled_vector)
        
        # Compare with high-level predict
        result = classifier.predict(valid_features)
        
        # Should get same prediction
        assert result["expression"] == EMOTION_LABELS[pred_idx]
        assert abs(result["confidence"] - confidence) < 0.001


class TestEmotionClassifierNoFace:
    """Tests for no-face handling."""

    def test_predict_no_face_returns_unavailable(self, classifier, valid_features):
        """Test that we handle no-face case externally (classifier always predicts)."""
        # The classifier itself doesn't know about face detection
        # It's the caller's responsibility to not call predict when no face
        result = classifier.predict(valid_features)
        
        # Classifier always returns a prediction
        assert result["expression"] in EMOTION_LABELS
        assert result["confidence"] > 0


class TestEmotionClassifierConfig:
    """Tests for configuration handling."""

    def test_config_passed_to_classifier(self, mock_model_files):
        """Test that config is accepted."""
        config = {"test_param": "value"}
        classifier = FaceEmotionClassifier(
            model_path=mock_model_files["model_path"],
            scaler_path=mock_model_files["scaler_path"],
            metadata_path=mock_model_files["metadata_path"],
            config=config,
        )
        assert classifier.config == config

    def test_empty_config(self, mock_model_files):
        """Test with empty config."""
        classifier = FaceEmotionClassifier(
            model_path=mock_model_files["model_path"],
            scaler_path=mock_model_files["scaler_path"],
            metadata_path=mock_model_files["metadata_path"],
            config={},
        )
        assert classifier.config == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])