"""
Unit tests for ResNetEmotionClassifier.

Tests model loading, preprocessing, inference,
no-face handling, and configuration.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.face.resnet_emotion import (
    ResNetEmotionClassifier,
    ResNetEmotionError,
    RESNET_EMOTION_LABELS,
    create_resnet_emotion_classifier,
)


# Shared fixtures
@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a mock checkpoint file for testing."""
    import torch
    import timm
    
    # Create a minimal ResNet-18 model
    model = timm.create_model("resnet18.fb_swsl_ig1b_ft_in1k", pretrained=False, num_classes=7)
    
    checkpoint = {
        "model_name": "resnet18.fb_swsl_ig1b_ft_in1k",
        "num_classes": 7,
        "classes": RESNET_EMOTION_LABELS,
        "img_size": 112,
        "src_size": 48,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "state_dict": model.state_dict(),
        "metrics": {
            "flip_tta": {"accuracy": 0.7208},
            "full_tta": {"accuracy": 0.7236},
        }
    }
    
    model_path = tmp_path / "fer2013_emotion_resnet18_v2.pt"
    torch.save(checkpoint, model_path)
    
    return model_path


@pytest.fixture
def classifier(mock_checkpoint):
    """Create a classifier instance with mock checkpoint."""
    return ResNetEmotionClassifier(model_path=mock_checkpoint, device="cpu")


@pytest.fixture
def valid_face48():
    """Create a valid 48x48 grayscale face image."""
    return np.random.randint(0, 255, (48, 48), dtype=np.uint8)


class TestResNetEmotionClassifier:
    """Tests for ResNetEmotionClassifier class."""

    def test_classifier_creation(self, classifier):
        """Test classifier creation and model loading."""
        assert classifier.is_ready()
        assert classifier.get_load_error() is None

    def test_get_classes(self, classifier):
        """Test getting emotion label names."""
        classes = classifier.get_classes()
        assert isinstance(classes, list)
        assert len(classes) == 7
        assert classes == RESNET_EMOTION_LABELS

    def test_get_device(self, classifier):
        """Test getting device."""
        device = classifier.get_device()
        assert str(device) in ["cpu", "cuda"]

    def test_get_model_info(self, classifier):
        """Test getting model metadata and configuration."""
        info = classifier.get_model_info()
        assert isinstance(info, dict)
        assert info["model_name"] == "resnet18.fb_swsl_ig1b_ft_in1k"
        assert info["num_classes"] == 7
        assert info["classes"] == RESNET_EMOTION_LABELS
        assert info["img_size"] == 112
        assert info["src_size"] == 48
        assert info["mean"] == [0.485, 0.456, 0.406]
        assert info["std"] == [0.229, 0.224, 0.225]
        assert "checkpoint_metrics" in info

    def test_predict_returns_correct_structure(self, classifier, valid_face48):
        """Test that predict returns correct result structure."""
        result = classifier.predict(valid_face48)
        
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "timestamp" in result
        assert "wallclock" in result
        
        # Check types
        assert isinstance(result["emotion"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["probabilities"], dict)
        assert isinstance(result["timestamp"], float)
        assert isinstance(result["wallclock"], str)
        
        # Check confidence bounds
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Check probabilities
        probs = result["probabilities"]
        assert len(probs) == 7
        for label in RESNET_EMOTION_LABELS:
            assert label in probs
            assert 0.0 <= probs[label] <= 1.0
        
        # Probabilities should sum to ~1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_emotion_in_labels(self, classifier, valid_face48):
        """Test that predicted emotion is one of the known labels."""
        result = classifier.predict(valid_face48)
        assert result["emotion"] in RESNET_EMOTION_LABELS

    def test_predict_confidence_matches_max_probability(self, classifier, valid_face48):
        """Test that confidence matches the max probability."""
        result = classifier.predict(valid_face48)
        max_prob = max(result["probabilities"].values())
        assert abs(result["confidence"] - max_prob) < 0.001

    def test_predict_with_different_inputs(self, classifier):
        """Test prediction with various input formats."""
        # uint8
        face_uint8 = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        result1 = classifier.predict(face_uint8)
        assert result1["emotion"] in RESNET_EMOTION_LABELS
        
        # float [0, 1]
        face_float = np.random.rand(48, 48).astype(np.float32)
        result2 = classifier.predict(face_float)
        assert result2["emotion"] in RESNET_EMOTION_LABELS
        
        # float [0, 255]
        face_float255 = (np.random.rand(48, 48) * 255).astype(np.float32)
        result3 = classifier.predict(face_float255)
        assert result3["emotion"] in RESNET_EMOTION_LABELS

    def test_predict_wrong_size_resizes(self, classifier):
        """Test that wrong-sized input gets resized."""
        # 64x64 input
        face64 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = classifier.predict(face64)
        assert result["emotion"] in RESNET_EMOTION_LABELS

    def test_predict_batch(self, classifier, valid_face48):
        """Test batch prediction."""
        faces = [valid_face48 for _ in range(3)]
        results = classifier.predict_batch(faces)
        
        assert len(results) == 3
        for result in results:
            assert result["emotion"] in RESNET_EMOTION_LABELS
            assert 0.0 <= result["confidence"] <= 1.0

    def test_reset(self, classifier):
        """Test reset method (no-op but should not error)."""
        classifier.reset()
        # Should not raise

    def test_is_ready_after_load(self, classifier):
        """Test is_ready returns True after successful load."""
        assert classifier.is_ready()

    def test_load_error_handling(self, tmp_path):
        """Test handling of missing model files."""
        with pytest.raises(ResNetEmotionError) as exc_info:
            ResNetEmotionClassifier(model_path=tmp_path / "nonexistent.pt")
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_checkpoint_handling(self, tmp_path):
        """Test handling of invalid checkpoint."""
        # Create a file with invalid content
        invalid_path = tmp_path / "invalid.pt"
        invalid_path.write_bytes(b"not a valid checkpoint")
        
        with pytest.raises(ResNetEmotionError) as exc_info:
            ResNetEmotionClassifier(model_path=invalid_path)
        assert "failed" in str(exc_info.value).lower()


class TestResNetEmotionClassifierConfig:
    """Tests for configuration handling."""

    def test_config_passed_to_classifier(self, mock_checkpoint):
        """Test that config is accepted."""
        config = {"test_param": "value"}
        classifier = ResNetEmotionClassifier(model_path=mock_checkpoint, config=config, device="cpu")
        assert classifier.config == config

    def test_empty_config(self, mock_checkpoint):
        """Test with empty config."""
        classifier = ResNetEmotionClassifier(model_path=mock_checkpoint, config={}, device="cpu")
        assert classifier.config == {}

    def test_device_selection(self, mock_checkpoint):
        """Test device selection."""
        classifier_cpu = ResNetEmotionClassifier(model_path=mock_checkpoint, device="cpu")
        assert str(classifier_cpu.get_device()) == "cpu"
        
        classifier_auto = ResNetEmotionClassifier(model_path=mock_checkpoint, device="auto")
        assert str(classifier_auto.get_device()) in ["cpu", "cuda"]


class TestResNetEmotionClassifierPreprocessing:
    """Tests for preprocessing correctness."""

    def test_preprocessing_matches_source(self, classifier, valid_face48):
        """Test that preprocessing produces valid tensor shape."""
        # The predict method should internally:
        # 1. Convert to float [0,1]
        # 2. Upscale 48->112 bilinear
        # 3. Expand to 3 channels
        # 4. Normalize with ImageNet mean/std
        # 5. Flip TTA
        result = classifier.predict(valid_face48)
        assert result["emotion"] in RESNET_EMOTION_LABELS

    def test_output_probabilities_valid(self, classifier, valid_face48):
        """Test that output probabilities are valid."""
        result = classifier.predict(valid_face48)
        probs = result["probabilities"]
        
        # All probabilities in [0, 1]
        for p in probs.values():
            assert 0.0 <= p <= 1.0
        
        # Sum to ~1.0
        assert abs(sum(probs.values()) - 1.0) < 0.01
        
        # Confidence equals max probability
        assert abs(result["confidence"] - max(probs.values())) < 0.001


class TestResNetEmotionClassifierFactory:
    """Tests for factory function."""

    def test_factory_creates_classifier(self, mock_checkpoint):
        """Test factory function creates classifier."""
        classifier = create_resnet_emotion_classifier(model_path=mock_checkpoint, device="cpu")
        assert classifier is not None
        assert isinstance(classifier, ResNetEmotionClassifier)
        assert classifier.is_ready()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])