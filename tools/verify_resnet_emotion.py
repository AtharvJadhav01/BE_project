#!/usr/bin/env python
"""
Standalone test script for ResNet-18 FER2013 emotion classifier.

Loads the pretrained model, runs inference on a test face image,
and prints the predicted emotion, confidence, and all class probabilities.

Usage:
    python tools/test_resnet_emotion.py [--image PATH] [--device auto|cpu|cuda]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.face.resnet_emotion import (
    ResNetEmotionClassifier,
    ResNetEmotionError,
    create_resnet_emotion_classifier,
    RESNET_EMOTION_LABELS,
)


def create_synthetic_face(emotion_hint: str = "neutral") -> np.ndarray:
    """
    Create a synthetic 48x48 grayscale face for testing.
    This is just for verification when no real image is available.
    """
    face = np.zeros((48, 48), dtype=np.uint8)

    # Simple synthetic face patterns for different emotions
    if emotion_hint == "happy":
        # Smile-like pattern
        face[15:35, 10:38] = 200
        face[25:30, 15:33] = 100  # mouth curve
    elif emotion_hint == "sad":
        face[15:35, 10:38] = 200
        face[28:33, 15:33] = 100  # downturned mouth
    elif emotion_hint == "surprise":
        face[10:30, 12:36] = 200  # wide eyes
        face[30:35, 18:30] = 100  # open mouth
    elif emotion_hint == "angry":
        face[12:25, 10:38] = 200  # furrowed brows
        face[28:33, 15:33] = 100
    else:
        # Neutral face
        face[15:33, 10:38] = 180
        face[20:28, 15:33] = 120

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, face.shape, dtype=np.uint8)
    face = cv2.add(face, noise)

    return face


def load_test_image(image_path: Path) -> np.ndarray:
    """Load and preprocess a test image to 48x48 grayscale."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Resize to 48x48 (the model's expected input size from face crop)
    face48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    return face48


def print_prediction(result: dict) -> None:
    """Print prediction results in a formatted way."""
    print(f"\nEmotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nProbabilities:")
    for label, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {label:10s}: {prob:.4f} {bar}")

    # Validate probabilities sum to ~1.0
    prob_sum = sum(result["probabilities"].values())
    print(f"\nProbability sum: {prob_sum:.6f} (should be ~1.0)")
    assert abs(prob_sum - 1.0) < 0.01, "Probabilities don't sum to 1!"

    # Validate all labels present
    assert set(result["probabilities"].keys()) == set(RESNET_EMOTION_LABELS), \
        "Missing or extra emotion labels"

    # Validate confidence matches max probability
    max_prob = max(result["probabilities"].values())
    assert abs(result["confidence"] - max_prob) < 0.001, \
        "Confidence doesn't match max probability"

    print("\n[OK] All validations passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Test ResNet-18 FER2013 emotion classifier"
    )
    parser.add_argument(
        "--image", "-i", type=Path, help="Path to test face image (will be resized to 48x48)"
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        help="Path to model checkpoint (default: models/emotion/fer2013_emotion_resnet18_v2.pt)",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run benchmark (100 inferences)",
    )
    parser.add_argument(
        "--synthetic",
        "-s",
        choices=RESNET_EMOTION_LABELS,
        help="Use synthetic face for specific emotion (for testing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet-18 FER2013 Emotion Classifier - Standalone Test")
    print("=" * 60)

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = Path(__file__).parent.parent / "models" / "emotion" / "fer2013_emotion_resnet18_v2.pt"

    print(f"\nModel: {model_path}")
    print(f"Device: {args.device}")

    # Load classifier
    print("\nLoading model...")
    start = time.perf_counter()
    try:
        classifier = create_resnet_emotion_classifier(
            model_path=model_path,
            device=args.device,
        )
        load_time = time.perf_counter() - start
        print(f"[OK] Model loaded in {load_time*1000:.1f} ms")
    except ResNetEmotionError as e:
        print(f"[FAIL] Failed to load model: {e}")
        return 1
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return 1

    # Print model info
    info = classifier.get_model_info()
    print(f"\nModel Info:")
    print(f"  Architecture: {info['model_name']}")
    print(f"  Classes: {info['classes']}")
    print(f"  Input size: {info['img_size']}x{info['img_size']}")
    print(f"  Source size: {info['src_size']}x{info['src_size']}")
    print(f"  Mean: {info['mean']}")
    print(f"  Std: {info['std']}")
    print(f"  Device: {info['device']}")
    if info["checkpoint_metrics"]:
        print(f"  Reported flip TTA accuracy: {info['checkpoint_metrics'].get('flip_tta', {}).get('accuracy', 'N/A')}")
        print(f"  Reported full TTA accuracy: {info['checkpoint_metrics'].get('full_tta', {}).get('accuracy', 'N/A')}")

    # Prepare test image
    if args.image:
        print(f"\nLoading test image: {args.image}")
        try:
            face48 = load_test_image(args.image)
            print(f"  Loaded and resized to 48x48")
        except Exception as e:
            print(f"✗ Failed to load image: {e}")
            return 1
    elif args.synthetic:
        print(f"\nGenerating synthetic face for: {args.synthetic}")
        face48 = create_synthetic_face(args.synthetic)
    else:
        print("\nNo image provided, using synthetic neutral face")
        face48 = create_synthetic_face("neutral")

    # Run inference
    print("\nRunning inference...")
    start = time.perf_counter()
    result = classifier.predict(face48)
    inference_time = time.perf_counter() - start
    print(f"Inference time: {inference_time*1000:.1f} ms")

    # Print results
    print_prediction(result)

    # Benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Benchmarking (100 inferences)...")
        print("=" * 60)

        times = []
        for i in range(100):
            start = time.perf_counter()
            _ = classifier.predict(face48)
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # ms
        print(f"\nResults (ms):")
        print(f"  Mean:   {times.mean():.2f}")
        print(f"  Median: {np.median(times):.2f}")
        print(f"  Min:    {times.min():.2f}")
        print(f"  Max:    {times.max():.2f}")
        print(f"  Std:    {times.std():.2f}")
        print(f"  FPS:    {1000/times.mean():.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())