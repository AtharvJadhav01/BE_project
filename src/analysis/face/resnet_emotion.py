"""
ResNet-18 FER2013 Emotion Classifier - Inference Component

Loads and runs the pretrained ResNet-18 model fine-tuned on FER2013.
This is a pure inference component - no training, no retraining.

Model: ResNet-18 (timm: resnet18.fb_swsl_ig1b_ft_in1k) fine-tuned on FER2013
Checkpoint: fer2013_emotion_resnet18_v2.pt (self-describing, contains preprocessing params)
Classes: angry, disgust, fear, happy, neutral, sad, surprise (7 classes)
Reported accuracy: 72.08% (flip TTA), 72.36% (full TTA)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm

from core.config import load_config
from core.logging import get_logger

logger = get_logger(__name__)

# Emotion labels from the pretrained checkpoint (verified from source)
# Note: "surprise" not "surprised" - matches checkpoint exactly
RESNET_EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class ResNetEmotionError(Exception):
    """Exception raised for ResNet emotion classifier errors."""
    pass


class ResNetEmotionClassifier:
    """
    Inference-only wrapper for the pretrained ResNet-18 FER2013 emotion classifier.

    Loads the pre-trained model checkpoint (which contains all preprocessing parameters),
    and provides expression estimation with confidence and probabilities.

    Key features:
    - Loads checkpoint only once
    - Uses torch.inference_mode() and model.eval()
    - Supports CPU and CUDA automatically
    - Configurable device selection
    - Handles missing/invalid checkpoints gracefully
    - Exact preprocessing reproduction from source repository
    - Flip TTA (Test-Time Augmentation) for improved accuracy
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the ResNet emotion classifier.

        Args:
            model_path: Path to the .pt checkpoint file
            config: Optional configuration dictionary
            device: Device to use ('cuda', 'cpu', or 'auto'). Default: 'auto'
        """
        self.config = config or {}
        self._model: Optional[torch.nn.Module] = None
        self._checkpoint: Dict[str, Any] = {}
        self._device: torch.device
        self._model_loaded: bool = False
        self._load_error: Optional[str] = None

        # Determine device
        if device is None or device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Determine model path
        if model_path is None:
            base_dir = Path(__file__).parent.parent.parent.parent / "models" / "emotion"
            model_path = base_dir / "fer2013_emotion_resnet18_v2.pt"

        self._model_path = model_path

        # Load model on initialization
        self._load_model()

    def _load_model(self) -> None:
        """Load the model checkpoint and rebuild architecture."""
        try:
            # Check file exists
            if not self._model_path.exists():
                raise ResNetEmotionError(f"Model file not found: {self._model_path}")

            # Load checkpoint (weights_only=True for security)
            logger.info("resnet_emotion_loading_checkpoint", path=str(self._model_path))
            start_time = time.perf_counter()
            self._checkpoint = torch.load(
                self._model_path, map_location="cpu", weights_only=True
            )
            load_time = time.perf_counter() - start_time
            logger.info("resnet_emotion_checkpoint_loaded", load_time_ms=load_time * 1000)

            # Extract preprocessing parameters from checkpoint (self-describing)
            self._model_name = self._checkpoint.get("model_name", "resnet18.fb_swsl_ig1b_ft_in1k")
            self._num_classes = self._checkpoint.get("num_classes", 7)
            self._classes = self._checkpoint.get("classes", RESNET_EMOTION_LABELS)
            self._img_size = self._checkpoint.get("img_size", 112)
            self._src_size = self._checkpoint.get("src_size", 48)
            self._mean = self._checkpoint.get("mean", [0.485, 0.456, 0.406])
            self._std = self._checkpoint.get("std", [0.229, 0.224, 0.225])

            # Verify class mapping
            if self._classes != RESNET_EMOTION_LABELS:
                logger.warning(
                    "resnet_emotion_class_mapping_differs",
                    expected=RESNET_EMOTION_LABELS,
                    actual=self._classes,
                )

            # Rebuild architecture and load weights
            logger.info("resnet_emotion_building_model", model_name=self._model_name)
            self._model = timm.create_model(
                self._model_name, pretrained=False, num_classes=self._num_classes
            )
            self._model.load_state_dict(self._checkpoint["state_dict"])
            self._model.eval().to(self._device)

            # Pre-compute normalization tensors on device
            self._mean_tensor = torch.tensor(self._mean, device=self._device).view(1, 3, 1, 1)
            self._std_tensor = torch.tensor(self._std, device=self._device).view(1, 3, 1, 1)

            self._model_loaded = True
            logger.info(
                "resnet_emotion_classifier_ready",
                device=str(self._device),
                model_name=self._model_name,
                num_classes=self._num_classes,
                classes=self._classes,
                img_size=self._img_size,
                src_size=self._src_size,
                reported_accuracy_flip_tta=self._checkpoint.get("metrics", {})
                .get("flip_tta", {})
                .get("accuracy"),
                reported_accuracy_full_tta=self._checkpoint.get("metrics", {})
                .get("full_tta", {})
                .get("accuracy"),
            )

        except ResNetEmotionError:
            raise
        except Exception as e:
            self._load_error = str(e)
            logger.error("resnet_emotion_load_failed", error=str(e))
            raise ResNetEmotionError(f"Failed to load ResNet emotion classifier: {e}")

    def is_ready(self) -> bool:
        """Check if the classifier is loaded and ready for inference."""
        return self._model_loaded and self._model is not None

    def get_load_error(self) -> Optional[str]:
        """Get the load error message if loading failed."""
        return self._load_error

    def get_device(self) -> torch.device:
        """Get the device being used for inference."""
        return self._device

    def get_classes(self) -> List[str]:
        """Get the emotion label names in order (from checkpoint)."""
        return self._classes.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and configuration."""
        return {
            "model_name": self._model_name,
            "num_classes": self._num_classes,
            "classes": self._classes,
            "img_size": self._img_size,
            "src_size": self._src_size,
            "mean": self._mean,
            "std": self._std,
            "device": str(self._device),
            "checkpoint_metrics": self._checkpoint.get("metrics", {}),
        }

    @torch.inference_mode()
    def predict(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Predict emotion from a preprocessed face image.

        Args:
            face_image: Grayscale face crop as numpy array (H, W), uint8 or float [0,1] or [0,255]
                       Expected to be SRC_SIZE x SRC_SIZE (48x48) from face detection crop

        Returns:
            Dictionary containing:
                - emotion: Predicted emotion label (str)
                - confidence: Classifier confidence 0-1 (float)
                - probabilities: Dict of all class probabilities
                - timestamp: Monotonic timestamp of prediction
                - wallclock: ISO format wallclock timestamp

        Raises:
            ResNetEmotionError: If model not loaded or inference fails
        """
        if not self.is_ready():
            raise ResNetEmotionError(
                f"Classifier not ready: {self._load_error or 'unknown error'}"
            )

        timestamp = time.monotonic()

        try:
            # Exact preprocessing from source repository (emotion_scanner.py predict function)
            # Input: uint8 48x48 grayscale face crop
            # Steps:
            # 1. Convert to float tensor [1, 1, 48, 48] in range [0, 1]
            # 2. Upscale to model input size (112x112) bilinear, align_corners=False
            # 3. Grayscale -> 3 channels (repeat)
            # 4. ImageNet normalization
            # 5. Flip TTA: original + horizontal flip, average predictions

            # Ensure uint8 input
            if face_image.dtype != np.uint8:
                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
                else:
                    face_image = face_image.astype(np.uint8)

            # Ensure correct size (SRC_SIZE x SRC_SIZE = 48x48)
            h, w = face_image.shape[:2]
            if h != self._src_size or w != self._src_size:
                # Resize if needed (should not happen if face detection crop is correct)
                import cv2
                face_image = cv2.resize(
                    face_image, (self._src_size, self._src_size), interpolation=cv2.INTER_AREA
                )

            # uint8 48x48 -> float tensor [1, 1, 48, 48] in range [0, 1]
            x = (
                torch.from_numpy(face_image)
                .to(self._device)
                .float()
                .div(255.0)
                .view(1, 1, self._src_size, self._src_size)
            )

            # Upscale to model input size exactly like in training
            # (bilinear, align_corners=False)
            x = F.interpolate(
                x, size=(self._img_size, self._img_size), mode="bilinear", align_corners=False
            )

            # Grayscale -> 3 channels + ImageNet normalization
            x = (x.expand(-1, 3, -1, -1) - self._mean_tensor) / self._std_tensor

            # Flip TTA: the original and its mirror image in one batch, predictions averaged
            x = torch.cat([x, torch.flip(x, dims=[3])])

            # Half precision on GPU for speed; full precision on CPU
            with torch.autocast(
                device_type=self._device.type, dtype=torch.float16, enabled=self._device.type == "cuda"
            ):
                logits = self._model(x)

            # Average flip TTA predictions and get probabilities
            probs = logits.float().softmax(dim=1).mean(dim=0).cpu().numpy()

            # Ensure valid probabilities
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else probs

            # Get predicted label and confidence
            pred_idx = int(np.argmax(probs))
            predicted_label = self._classes[pred_idx] if pred_idx < len(self._classes) else "unknown"
            confidence = float(probs[pred_idx])

            # Build probabilities dict
            probabilities = {
                self._classes[i]: float(probs[i]) for i in range(len(self._classes))
            }

            # Generate wallclock timestamp
            from datetime import datetime, timezone

            wallclock = (
                datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )

            result = {
                "emotion": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "timestamp": timestamp,
                "wallclock": wallclock,
            }

            logger.debug(
                "resnet_emotion_prediction",
                emotion=predicted_label,
                confidence=confidence,
            )

            return result

        except ResNetEmotionError:
            raise
        except Exception as e:
            logger.error("resnet_emotion_prediction_failed", error=str(e))
            raise ResNetEmotionError(f"Prediction failed: {e}")

    def predict_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Predict emotions for a batch of face images.

        Args:
            face_images: List of grayscale face crops (each H, W), uint8

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in face_images]

    def reset(self) -> None:
        """Reset classifier state (no-op for stateless classifier)."""
        pass


def create_resnet_emotion_classifier(
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> ResNetEmotionClassifier:
    """
    Factory function to create a ResNetEmotionClassifier.

    Args:
        config: Optional configuration dictionary
        model_path: Optional path to model checkpoint
        device: Device to use ('cuda', 'cpu', 'auto')

    Returns:
        Initialized ResNetEmotionClassifier instance
    """
    return ResNetEmotionClassifier(model_path=model_path, config=config, device=device)