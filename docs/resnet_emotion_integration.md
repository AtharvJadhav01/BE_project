# ResNet-18 FER2013 Emotion Classifier Integration

## Overview

This document describes the integration of a pretrained ResNet-18 model fine-tuned on the FER2013 dataset for facial expression recognition into the existing BE Project facial analysis pipeline.

## Model Details

| Property | Value |
|----------|-------|
| **Model** | ResNet-18 (timm: `resnet18.fb_swsl_ig1b_ft_in1k`) |
| **Parameters** | 11.2M |
| **Training Data** | FER2013 (35,887 images, 48×48 grayscale) |
| **Classes** | 7 (angry, disgust, fear, happy, neutral, sad, surprise) |
| **Checkpoint** | `fer2013_emotion_resnet18_v2.pt` (43.8 MB) |
| **License** | MIT |
| **Source** | https://github.com/TyomPapiyan/fer2013-emotion-scanner |

## Reported Accuracy (from source)

| Evaluation | Test Accuracy |
|------------|---------------|
| Single forward pass (no TTA) | 70.80% |
| Flip TTA (used in scanner) | 72.08% |
| Full TTA (6 crops × 2 flips) | 72.36% |

Per-class F1 scores (Full TTA):
- Happy: 0.902
- Surprise: 0.830
- Disgust: 0.711
- Angry: 0.654
- Neutral: 0.691
- Sad: 0.603
- Fear: 0.576

## Architecture

### Preprocessing Pipeline (exact reproduction from source)

```
Webcam frame (BGR)
    ↓
MediaPipe FaceLandmarker → bounding box (x, y, w, h)
    ↓
Square crop with 15% margin (FACE_MARGIN = 1.15)
    ↓
Convert to grayscale
    ↓
Resize to 48×48 (SRC_SIZE) using INTER_AREA
    ↓
Normalize to [0, 1] (divide by 255)
    ↓
Bilinear upscale to 112×112 (IMG_SIZE) with align_corners=False
    ↓
Repeat to 3 channels
    ↓
ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ↓
Flip TTA: original + horizontal flip → average predictions
    ↓
Softmax → 7-class probabilities
```

### Key Design Decisions

1. **Self-describing checkpoint**: All preprocessing parameters stored in the `.pt` file
2. **Flip TTA**: Horizontal flip test-time augmentation for +1.3% accuracy
3. **No face alignment**: Uses MediaPipe bounding box directly (matches training)
4. **CPU/CUDA support**: Automatic device detection, half-precision on GPU

## Integration Architecture

```
Webcam (640×480 @ 15 FPS)
    ↓
FaceCapture (src/face/capture.py)
    ↓
MediaPipe FaceLandmarker (468 landmarks, 52 blendshapes, head pose)
    ├─→ Existing Pipeline (MediaPipe features → Random Forest → expression_estimate)
    │       ↓
    │   face_expressions.jsonl (27 features + Random Forest)
    │
    └─→ ResNet-18 Pipeline (NEW)
            ↓
        Face crop from bounding box (15% margin, grayscale, 48×48)
            ↓
        ResNet-18 FER2013 inference
            ↓
        face_resnet_emotions.jsonl (7-class probabilities)
```

## Configuration

Add to `config.yaml` under `analysis.face`:

```yaml
analysis:
  face:
    # ... existing settings ...
    
    # ResNet-18 FER2013 pretrained emotion classifier
    resnet_emotion_enabled: true
    resnet_emotion_model_path: "models/emotion/fer2013_emotion_resnet18_v2.pt"
    resnet_emotion_device: "auto"  # auto, cuda, cpu
    resnet_emotion_confidence_threshold: 0.5
```

## Files Created/Modified

### New Files
- `src/analysis/face/resnet_emotion.py` - ResNet emotion classifier module
- `tools/test_resnet_emotion.py` - Standalone test/benchmark script
- `tests/unit/test_resnet_emotion.py` - Unit tests (20 tests)
- `models/emotion/fer2013_emotion_resnet18_v2.pt` - Model checkpoint (copied)

### Modified Files
- `src/core/config.py` - Added ResNet config to `AnalysisFaceConfig`
- `config/config.yaml` - Added ResNet emotion settings
- `src/face/capture.py` - Integrated ResNet classifier into analysis pipeline
- `src/ui/main_window.py` - Added ResNet emotion display to FaceAnalysisPanel

## Output Files (per session)

Each session now generates these analysis files in `data/sessions/<session_id>/analysis/`:

| File | Content |
|------|---------|
| `face_features.jsonl` | Normalized MediaPipe features (existing) |
| `face_states.jsonl` | Rule-based facial states (existing) |
| `face_expressions.jsonl` | Random Forest expression estimates (existing) |
| `face_resnet_emotions.jsonl` | **NEW** ResNet-18 FER2013 probabilities |

### ResNet Emotions JSONL Format

```json
{
  "timestamp_monotonic": 12345.678,
  "timestamp_wallclock": "2026-10-08T00:00:00.000Z",
  "frame_index": 42,
  "emotion": "happy",
  "confidence": 0.874,
  "probabilities": {
    "angry": 0.010,
    "disgust": 0.002,
    "fear": 0.020,
    "happy": 0.874,
    "neutral": 0.030,
    "sad": 0.040,
    "surprise": 0.024
  },
  "baseline_ready": true
}
```

Note: Uses "surprise" (not "surprised") to match checkpoint exactly.

## GUI Integration

The FaceAnalysisPanel now displays two emotion estimates:

1. **Estimated Expression** (existing): Random Forest on MediaPipe features
2. **ResNet-18 Emotion (FER2013)** (new): Image-based ResNet classification

## Performance Benchmarks

### Model Loading
- CUDA: ~236 ms
- CPU: ~167 ms

### Inference Latency (steady state, 100 runs)

| Device | Mean | Median | Min | Max | FPS |
|--------|------|--------|-----|-----|-----|
| CUDA (RTX 30xx) | 3.2 ms | 3.1 ms | 2.3 ms | 7.2 ms | 309 |
| CPU (Intel i7) | 20.5 ms | 21.1 ms | 16.0 ms | 24.0 ms | 49 |

### Comparison with Existing Classifier

| Aspect | Random Forest (existing) | ResNet-18 (new) |
|--------|--------------------------|-----------------|
| Input | 27 MediaPipe features | 48×48 face image |
| Model Size | ~9 MB (joblib) | 43.8 MB (.pt) |
| FER2013 Accuracy | ~49.14% | 72.08% (flip TTA) |
| Inference (CPU) | <1 ms | ~20 ms |
| Inference (CUDA) | N/A | ~3 ms |
| Features | Landmarks + blendshapes | Raw pixels |

## Usage

### Standalone Test
```bash
# Test with synthetic face
python tools/test_resnet_emotion.py --synthetic happy

# Test with image file
python tools/test_resnet_emotion.py --image path/to/face.jpg

# Benchmark
python tools/test_resnet_emotion.py --synthetic neutral --benchmark
```

### In Code
```python
from analysis.face.resnet_emotion import create_resnet_emotion_classifier

# Create classifier (loads once)
classifier = create_resnet_emotion_classifier(device="auto")

# Predict from 48x48 grayscale face crop
face_crop = ...  # uint8 numpy array (48, 48)
result = classifier.predict(face_crop)

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probs: {result['probabilities']}")
```

### Enable/Disable at Runtime
```python
face_capture.set_emotion_enabled(False)  # Disables both classifiers
# ResNet specifically controlled via config.resnet_emotion_enabled
```

## Limitations

1. **Frontal faces only**: Accuracy degrades with pose >30°
2. **Lighting sensitive**: Requires decent illumination
3. **Single face**: Only processes largest detected face
4. **Expression ≠ Emotion**: Classifies facial expressions, not internal feelings
5. **Disgust class**: Trained on only 436 images (least reliable)
6. **Fear/Sad confusion**: Common confusion between fear, sad, neutral

## Testing

Run unit tests:
```bash
pytest tests/unit/test_resnet_emotion.py -v
```

All 20 tests pass covering:
- Model loading
- Preprocessing
- Inference
- Error handling
- Configuration
- Device selection

## Future Improvements

- [ ] ONNX export for faster CPU inference
- [ ] Multi-face support (process all detected faces)
- [ ] Temporal smoothing for ResNet predictions
- [ ] Fusion of Random Forest + ResNet predictions
- [ ] Evaluation on local FER2013 test set
- [ ] Face alignment before cropping (improve accuracy)

## References

- FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
- Source Repository: https://github.com/TyomPapiyan/fer2013-emotion-scanner
- timm ResNet-18 SWSL: https://github.com/huggingface/pytorch-image-models
- YuNet Face Detector: https://github.com/opencv/opencv_zoo