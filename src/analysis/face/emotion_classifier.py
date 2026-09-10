"""
Face Emotion Classifier - Inference Component

Loads and runs the Phase 3 Random Forest classifier for facial expression estimation.
This is a pure inference component - no training, no retraining.

Uses the validated Phase 3 model (rf_emotion_200est_15depth) with 27 features.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from core.config import load_config
from core.logging import get_logger

logger = get_logger(__name__)

# Phase 3 model feature names (exact order from training metadata)
MODEL_FEATURE_NAMES = [
    "eye_openness_left",
    "eye_openness_right",
    "eye_openness_avg",
    "eye_aspect_ratio_left",
    "eye_aspect_ratio_right",
    "blink_rate",
    "mouth_openness",
    "mouth_width",
    "lip_press",
    "smile_asymmetry",
    "mouth_funnel",
    "mouth_pucker",
    "mouth_stretch_left",
    "mouth_stretch_right",
    "brow_inner_up",
    "brow_down_left",
    "brow_down_right",
    "brow_outer_up_left",
    "brow_outer_up_right",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "facial_movement_velocity",
    "facial_symmetry",
    "overall_expressivity",
    "nose_sneer_left",
    "nose_sneer_right",
]

# Emotion labels (matching FER2013 label map)
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
    "neutral",
]


class EmotionClassifierError(Exception):
    """Exception raised for emotion classifier errors."""
    pass


class FaceEmotionClassifier:
    """
    Inference-only wrapper for the Phase 3 Random Forest emotion classifier.
    
    Loads the pre-trained model and scaler, validates feature compatibility,
    and provides expression estimation with confidence and probabilities.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the emotion classifier.
        
        Args:
            model_path: Path to the .joblib model file
            scaler_path: Path to the scaler .joblib file
            metadata_path: Path to the metadata JSON file
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._model: Any = None
        self._scaler: Any = None
        self._metadata: Dict[str, Any] = {}
        self._feature_names: List[str] = MODEL_FEATURE_NAMES
        self._label_names: List[str] = EMOTION_LABELS
        self._model_loaded: bool = False
        self._load_error: Optional[str] = None
        
        # Determine model paths
        if model_path is None:
            base_dir = Path(__file__).parent.parent.parent.parent / "training" / "models"
            model_path = base_dir / "rf_emotion_200est_15depth.joblib"
            scaler_path = base_dir / "scaler.joblib"
            metadata_path = base_dir / "rf_emotion_200est_15depth_metadata.json"
        
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._metadata_path = metadata_path
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model, scaler, and metadata from disk."""
        try:
            # Check files exist
            if not self._model_path.exists():
                raise EmotionClassifierError(f"Model file not found: {self._model_path}")
            if not self._scaler_path.exists():
                raise EmotionClassifierError(f"Scaler file not found: {self._scaler_path}")
            if not self._metadata_path.exists():
                raise EmotionClassifierError(f"Metadata file not found: {self._metadata_path}")
            
            # Load model
            self._model = joblib.load(self._model_path)
            logger.info("emotion_model_loaded", path=str(self._model_path))
            
            # Load scaler
            self._scaler = joblib.load(self._scaler_path)
            logger.info("emotion_scaler_loaded", path=str(self._scaler_path))
            
            # Load metadata
            import json
            with open(self._metadata_path, "r") as f:
                self._metadata = json.load(f)
            
            # Validate feature names match
            saved_features = self._metadata.get("feature_names", [])
            if saved_features != self._feature_names:
                logger.warning(
                    "feature_names_mismatch",
                    expected=self._feature_names,
                    actual=saved_features
                )
                # Use saved feature names for compatibility
                self._feature_names = saved_features
            
            # Validate feature dimension
            feature_dim = self._metadata.get("feature_dim", len(self._feature_names))
            if feature_dim != len(self._feature_names):
                raise EmotionClassifierError(
                    f"Feature dimension mismatch: expected {len(self._feature_names)}, "
                    f"metadata says {feature_dim}"
                )
            
            # Validate model has predict_proba
            if not hasattr(self._model, "predict_proba"):
                raise EmotionClassifierError(
                    "Loaded model does not support predict_proba"
                )
            
            # Validate label names
            target_names = self._metadata.get("target_names", [])
            if target_names:
                self._label_names = target_names
            
            self._model_loaded = True
            logger.info(
                "emotion_classifier_ready",
                model_type=self._metadata.get("model_type"),
                feature_dim=feature_dim,
                num_classes=len(self._label_names),
                test_accuracy=self._metadata.get("metrics", {}).get("test", {}).get("accuracy")
            )
            
        except EmotionClassifierError:
            raise
        except Exception as e:
            self._load_error = str(e)
            logger.error("emotion_classifier_load_failed", error=str(e))
            raise EmotionClassifierError(f"Failed to load emotion classifier: {e}")
    
    def is_ready(self) -> bool:
        """Check if the classifier is loaded and ready for inference."""
        return self._model_loaded and self._model is not None and self._scaler is not None
    
    def get_load_error(self) -> Optional[str]:
        """Get the load error message if loading failed."""
        return self._load_error
    
    def get_feature_names(self) -> List[str]:
        """Get the expected feature names in order."""
        return self._feature_names.copy()
    
    def get_label_names(self) -> List[str]:
        """Get the emotion label names in order."""
        return self._label_names.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self._metadata.copy()
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present.
        
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = [name for name in self._feature_names if name not in features]
        return len(missing) == 0, missing
    
    def prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare feature vector from dictionary, ensuring correct ordering.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            2D numpy array (1, n_features) ready for scaling
            
        Raises:
            EmotionClassifierError: If features are missing or invalid
        """
        is_valid, missing = self.validate_features(features)
        if not is_valid:
            raise EmotionClassifierError(f"Missing required features: {missing}")
        
        # Build feature vector in exact model order
        feature_vector = np.array(
            [features[name] for name in self._feature_names],
            dtype=np.float32
        ).reshape(1, -1)
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(feature_vector)):
            raise EmotionClassifierError("Feature vector contains NaN or Inf values")
        
        return feature_vector
    
    def predict(
        self,
        features: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predict estimated expression from facial features.
        
        Args:
            features: Dictionary of facial features (must contain all required features)
            timestamp: Optional monotonic timestamp for the prediction
            
        Returns:
            Dictionary containing:
                - expression: Estimated expression label (str)
                - confidence: Classifier confidence 0-1 (float)
                - probabilities: Dict of all class probabilities
                - timestamp: Monotonic timestamp of prediction
                - wallclock: ISO format wallclock timestamp
                - model_version: Model metadata timestamp
                
        Raises:
            EmotionClassifierError: If model not loaded or inference fails
        """
        if not self.is_ready():
            raise EmotionClassifierError(
                f"Classifier not ready: {self._load_error or 'unknown error'}"
            )
        
        if timestamp is None:
            timestamp = time.monotonic()
        
        try:
            # Prepare and scale feature vector
            feature_vector = self.prepare_feature_vector(features)
            feature_vector_scaled = self._scaler.transform(feature_vector)
            
            # Get prediction and probabilities
            pred_idx = self._model.predict(feature_vector_scaled)[0]
            probas = self._model.predict_proba(feature_vector_scaled)[0]
            
            # Ensure probabilities are valid
            probas = np.clip(probas, 0.0, 1.0)
            probas = probas / np.sum(probas) if np.sum(probas) > 0 else probas
            
            # Get predicted label and confidence
            predicted_label = self._label_names[pred_idx] if pred_idx < len(self._label_names) else "unknown"
            confidence = float(probas[pred_idx])
            
            # Build probabilities dict
            probabilities = {
                self._label_names[i]: float(probas[i])
                for i in range(len(self._label_names))
            }
            
            # Generate wallclock timestamp
            from datetime import datetime, timezone
            wallclock = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
            
            result = {
                "expression": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "timestamp": timestamp,
                "wallclock": wallclock,
                "model_version": self._metadata.get("timestamp", "unknown"),
            }
            
            logger.debug(
                "emotion_prediction",
                expression=predicted_label,
                confidence=confidence
            )
            
            return result
            
        except EmotionClassifierError:
            raise
        except Exception as e:
            logger.error("emotion_prediction_failed", error=str(e))
            raise EmotionClassifierError(f"Prediction failed: {e}")
    
    def predict_raw(self, feature_vector: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Low-level prediction from pre-scaled feature vector.
        
        Used for testing and validation.
        
        Args:
            feature_vector: Pre-scaled 2D feature array (1, n_features)
            
        Returns:
            Tuple of (predicted_index, confidence, probabilities_array)
        """
        if not self.is_ready():
            raise EmotionClassifierError("Classifier not ready")
        
        pred_idx = self._model.predict(feature_vector)[0]
        probas = self._model.predict_proba(feature_vector)[0]
        confidence = float(probas[pred_idx])
        
        return int(pred_idx), confidence, probas
    
    def reset(self) -> None:
        """Reset classifier state (no-op for stateless classifier)."""
        pass


def create_emotion_classifier(config: Optional[Dict[str, Any]] = None) -> FaceEmotionClassifier:
    """
    Factory function to create a FaceEmotionClassifier.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized FaceEmotionClassifier instance
    """
    return FaceEmotionClassifier(config=config)