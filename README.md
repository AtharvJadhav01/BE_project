# Multimodal Human State Capture Application

## Overview

A desktop application designed to capture and synchronize multiple human input modalities during a user session:

- **Voice / Audio** — Microphone capture via `sounddevice`, WAV storage with RMS/voice activity
- **Keyboard / Keystrokes** — Global capture via `pynput`, JSONL events with sensitive-field redaction
- **Facial Expressions** — Webcam + MediaPipe Face Mesh (468 landmarks, 52 blendshapes, head pose, iris), JSONL feature storage

All modalities are timestamped with a shared monotonic clock and stored in a session-based directory structure for reproducible analysis.

## Features

- 🎙️ **Voice/audio capture** — 16kHz mono WAV chunks, real-time pitch/energy/spectral analysis
- ⌨️ **Keystroke event capture** — Global hook via `pynput`, sensitive-field redaction, context capture
- 📷 **Webcam capture** — 640×480 @ 15fps via OpenCV + MediaPipe Face Mesh
- 🙂 **Facial feature & expression extraction** — 468 3D landmarks, 52 blendshapes, head pose, iris tracking
- 🕒 **Timestamp synchronization** — Monotonic clock for alignment, wall-clock for metadata
- 📊 **Structured session-based storage** — WAV chunks, JSONL events, unified event log
- ⚙️ **Configurable capture settings** — YAML config, per-modality enable/disable
- 🖥️ **Desktop UI (PySide6)** — Real-time face/audio/keyboard analysis panels, multimodal fusion display
- 🧩 **Modular architecture** — Per-modality capture/process/store, pluggable for new modalities
- 🧪 **ML benchmarking suite** — FER2013 benchmark with LinearSVC, LogisticRegression, XGBoost, PCA+SVM, CNN
- 🔒 **Privacy-aware** — Local-only, sensitive-field redaction, no raw video storage, explicit consent

## Verified Benchmark Results (FER2013 PublicTest — 1,162 face-detected samples)

| Model | Features | Test Accuracy | Macro F1 | Weighted F1 |
|-------|----------|---------------|----------|-------------|
| **LinearSVC** (protected benchmark) | 1,449 (Landmarks + Blendshapes) | **50.69%** | **49.19%** | **49.98%** |
| LogisticRegression | 1,457 (Expanded) | 49.14% | 48.61% | 49.12% |
| Geometric + LinearSVC* | 1,503 (Combined) | 50.60% | 49.29% | 49.81% |
| XGBoost | 1,449 | 46.90% | 45.55% | 45.87% |
| PCA + RBF SVM | 1,449 → 100 PCA | 46.73% | 45.72% | 46.07% |
| CNN (SimpleCNN, raw 48×48) | Raw pixels | 19.28% | 16.46% | 16.45% |

\* Geometric combined used different validation split (1,115/124/1,162); not directly comparable to protected 1,054/185/1,162 split.

## Webcam Evaluation Results (Held-out Real-World Dataset)

| Model | Features | Test Accuracy | Macro F1 | Weighted F1 |
|-------|----------|---------------|----------|-------------|
| **LinearSVC** (same as FER2013) | 1,449 (Landmarks + Blendshapes) | **18.79%** | **8.62%** | **9.69%** |

**Webcam Dataset:** 1,804 images (7 classes: angry: 326, disgust: 178, fear: 331, happy: 234, sad: 249, surprised: 267, neutral: 219)

**Domain Shift:** A substantial domain shift was observed. Performance on the held-out webcam dataset was substantially lower than on FER2013. This demonstrates that benchmark performance does not directly translate to real-world webcam performance.

| Dataset | Accuracy | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| FER2013 PublicTest | 50.69% | 49.19% | 49.98% |
| Webcam Test Set | 18.79% | 8.62% | 9.69% |

**Gap:** −31.90 pp accuracy, −40.57 pp Macro F1, −40.29 pp Weighted F1.

A substantial domain shift was observed. Performance on the held-out webcam dataset was substantially lower than on FER2013. This demonstrates that benchmark performance does not directly translate to real-world webcam performance.

The webcam dataset contains 1,804 manually labeled images across 7 classes. The webcam dataset is not part of FER2013 training and is used as an external/held-out real-world evaluation set. The webcam result should NOT be presented as the FER2013 benchmark accuracy. Confidence scores must not be interpreted as accuracy.

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run desktop application
python src/main.py

# Run FER2013 benchmark (reproduces all verified results)
python training/benchmark.py

# Train CNN on FER2013 raw images
python training/train_cnn.py

# Collect webcam expression dataset (for real-world evaluation)
python tools/webcam_dataset_collector.py

# Validate webcam dataset
python tools/validate_webcam_dataset.py --dataset data/webcam_expression_dataset

# Evaluate best model on webcam data
python training/evaluate_webcam.py
```

## Project Structure

```
BE_Project/
├── SPEC.md                    # Specification (source of truth)
├── README.md                  # This file
├── config/
│   └── config.yaml            # All runtime settings
├── docs/
│   ├── project_status.md      # Complete project status (this audit)
│   ├── facial_expression_benchmark_report.md
│   ├── facial_expression_optimization_report.md
│   ├── pca_rbf_svm_experiment_report.md
│   ├── xgboost_experiment_report.md
│   ├── geometric_features_experiment_report.md
│   ├── cnn_fer2013_experiment_report.md
│   ├── real_world_face_validation.md
│   ├── webcam_dataset_protocol.md
│   ├── ARCHITECTURE.md
│   ├── DATA_SCHEMA.md
│   ├── PRIVACY.md
│   ├── LIBRARIES.md
│   └── REQUIREMENTS.md
├── src/
│   ├── main.py
│   ├── core/                  # Config, events, logging, timestamps
│   ├── audio/                 # Audio capture, processing, analysis
│   ├── keyboard/              # Keyboard capture, processing, analysis
│   ├── face/                  # Face capture, MediaPipe, analysis
│   ├── analysis/              # Face, audio, keyboard, multimodal analysis
│   ├── processing/            # Synchronization
│   ├── session/               # Session management
│   ├── storage/               # Session persistence
│   └── ui/                    # PySide6 UI
├── training/                  # ML experiments, FER2013 benchmarks
├── tools/                     # Webcam dataset tools
├── data/
│   ├── fer2013.csv            # FER2013 dataset
│   └── sessions/              # Captured session data
├── config/config.yaml
├── requirements.txt
└── pyproject.toml
```

## Data Flow

```text
Voice ───────────┐
                 │
Keyboard ────────┼──> Timestamping ──> Processing ──> Storage
                 │
Face ────────────┘
```

All modalities share a common monotonic timestamp generator for precise cross-modal alignment.

## Verified Model Artifacts (FER2013)

| Model | Path | Status |
|-------|------|--------|
| LinearSVC (1,449 feat) | `training/models/linear_svc_optimized_1449.joblib` | ✅ |
| LinearSVC Scaler | `training/models/scaler_optimized_1449.joblib` | ✅ |
| LinearSVC Metadata | `training/models/linear_svc_optimized_1449_metadata.json` | ✅ |
| LogisticRegression (1,457) | `training/models/logistic_regression_expanded_1457.joblib` | ✅ |
| XGBoost | `training/models/xgboost_optimized_1449.joblib` | ✅ |
| PCA + RBF SVM | `training/models/pca_rbf_svm_optimized.joblib` | ✅ |
| CNN (SimpleCNN) | `training/models/cnn_fer2013_best.pt` | ✅ |
| CNN Metadata | `training/models/cnn_fer2013_metadata.json` | ✅ |
| CNN Benchmark Split | `training/cnn_benchmark_split.json` | ✅ |
| **ResNet-18 (FER2013)** | `models/emotion/fer2013_emotion_resnet18_v2.pt` | ✅ |

All artifacts include metadata with hyperparameters, seeds, split info; reload verification passed for all.

## ResNet-18 Emotion Recognition

### 1. Overview

The application includes a pretrained **ResNet-18** model fine-tuned on the **FER2013** dataset for image-based facial emotion recognition. This runs alongside the existing MediaPipe facial analysis pipeline (landmarks, blendshapes, Random Forest classifier), providing an additional emotion estimation path.

**High-level flow:**

```text
Webcam (640×480 @ 15 FPS)
    ↓
MediaPipe FaceLandmarker → face detection + bounding box
    ↓
Face crop with 15% margin → grayscale → 48×48 resize
    ↓
ResNet-18 FER2013 inference (Flip TTA)
    ↓
Emotion prediction (7 classes) + probabilities
    ↓
GUI display (FaceAnalysisPanel) + session storage (face_resnet_emotions.jsonl)
```

The existing pipeline (27 MediaPipe features → Random Forest → `expression_estimate`) remains fully active. ResNet-18 provides a separate `resnet_emotion_estimate` based on raw face pixels.

### 2. Model

| Property | Value |
|----------|-------|
| **File** | `models/emotion/fer2013_emotion_resnet18_v2.pt` |
| **Architecture** | ResNet-18 |
| **timm architecture** | `resnet18.fb_swsl_ig1b_ft_in1k` |
| **Parameter count** | ~11.2M |
| **Classes (7)** | `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` |

> The model is tracked using **Git LFS** (see `.gitattributes`). Users cloning the repository must have Git LFS installed so the actual model file is downloaded.

### 3. Prerequisites

- **Python 3.11+** — virtual environment recommended
- **Git** — for cloning
- **Git LFS** — required to download the model file
- **Project dependencies** — see `requirements.txt`
- **CUDA-capable GPU (optional)** — the implementation auto-detects CUDA; CPU fallback is supported

### 4. Clone and Git LFS Setup

```cmd
git clone <repository-url>
cd app_work
git lfs install
git lfs pull
```

Verify the model was downloaded:
```cmd
dir models\emotion\fer2013_emotion_resnet18_v2.pt
```

### 5. Python Environment Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 6. Verify the ResNet Model

Run the standalone verification script:

```cmd
python tools/verify_resnet_emotion.py --synthetic neutral
```

**What it checks:**
- Model loads successfully from `models/emotion/fer2013_emotion_resnet18_v2.pt`
- Architecture matches `resnet18.fb_swsl_ig1b_ft_in1k`
- CUDA is detected (falls back to CPU if unavailable)
- Preprocessing pipeline matches the source repository exactly
- Inference produces valid probabilities summing to 1.0
- All 7 emotion labels are present in output

Optional flags:
- `--image PATH` — test with a custom face image
- `--device auto|cpu|cuda` — force device selection
- `--benchmark` — run 100 inferences and report latency statistics

### 7. Run the Application

From the `app_work` directory:

```cmd
set PYTHONPATH=%CD%\src
python -m src.main
```

**Expected behavior:**
1. Application window opens with Face, Audio, Keyboard analysis panels
2. Click **Start Session** → webcam initializes (640×480 @ 15 FPS)
3. Face is detected via MediaPipe FaceLandmarker
4. ResNet-18 inference runs on each detected face crop
5. **ResNet-18 Emotion (FER2013)** panel updates in real time:
   - Emotion label (e.g., `happy`, `neutral`, `surprise`)
   - Confidence percentage
6. Results stored in session directory under `analysis/face_resnet_emotions.jsonl`

### 8. Configuration

ResNet settings are in `config/config.yaml` under `analysis.face`:

```yaml
analysis:
  face:
    # ... existing settings ...
    
    # ResNet-18 FER2013 pretrained emotion classifier
    resnet_emotion_enabled: true           # enable/disable ResNet pipeline
    resnet_emotion_model_path: "models/emotion/fer2013_emotion_resnet18_v2.pt"
    resnet_emotion_device: "auto"          # auto, cuda, cpu
    resnet_emotion_confidence_threshold: 0.5
```

### 9. Output / Session Data

Per session, ResNet results are written to:
```
data/sessions/<session_id>/analysis/face_resnet_emotions.jsonl
```

Each line is a JSON record:

```json
{
  "timestamp_monotonic": 12345.678,
  "timestamp_wallclock": "2026-10-08T00:00:00.000Z",
  "frame_index": 42,
  "emotion": "happy",
  "confidence": 0.874,
  "probabilities": {
    "angry": 0.010, "disgust": 0.002, "fear": 0.020,
    "happy": 0.874, "neutral": 0.030, "sad": 0.040, "surprise": 0.024
  },
  "baseline_ready": true
}
```

> Note: Uses `surprise` (not `surprised`) to match the checkpoint exactly.

### 10. Testing

Run the unit test suite:

```cmd
pytest tests/unit -q
```

**Current verified result:** 293 tests passed (includes 20 ResNet-specific tests in `tests/unit/test_resnet_emotion.py`).

### 11. Troubleshooting

| Issue | Cause / Resolution |
|-------|-------------------|
| `Model file not found` | Git LFS not installed or `git lfs pull` not run. Re-run `git lfs install && git lfs pull`. |
| `weights_only load failed` | PyTorch ≥2.6 changed default `weights_only=True`. The implementation handles this; ensure model file is intact. |
| `CUDA not available` / inference slow | No NVIDIA GPU or drivers. CPU fallback is automatic. Install CUDA toolkit for GPU acceleration. |
| Webcam not detected | Check `camera.device` in config (default `0`). Ensure no other app is using the camera. |
| `ModuleNotFoundError` / import errors | Run from `app_work` directory with `set PYTHONPATH=%CD%\src` before `python -m src.main`. |
| ResNet panel shows `--` / 0% | Face not detected, or `resnet_emotion_enabled: false` in config. Check logs for `resnet_emotion_initializing` / `resnet_inference_executed`. |

For deeper technical details (preprocessing pipeline, benchmark comparisons, integration architecture), see:
[`docs/resnet_emotion_integration.md`](docs/resnet_emotion_integration.md)

## Privacy

- **Local-only** — No network communication, no telemetry
- **Explicit consent** — Session start requires explicit button press
- **Sensitive redaction** — Password fields detected via OS accessibility; keys redacted
- **No raw video** — Only derived facial features stored
- **Explicit modality control** — Each modality independently enable/disable
- **Session deletion** — Full session removal via UI or filesystem

## Documentation

| Document | Description |
|----------|-------------|
| `SPEC.md` | Primary specification (SDD) |
| `docs/project_status.md` | Complete project status & audit |
| `docs/facial_expression_benchmark_report.md` | Full FER2013 benchmark audit |
| `docs/facial_expression_optimization_report.md` | LinearSVC optimization |
| `docs/pca_rbf_svm_experiment_report.md` | PCA + RBF SVM |
| `docs/xgboost_experiment_report.md` | XGBoost experiment |
| `docs/geometric_features_experiment_report.md` | Geometric features |
| `docs/cnn_fer2013_experiment_report.md` | CNN experiment |
| `docs/real_world_face_validation.md` | Webcam evaluation protocol |
| `docs/webcam_dataset_protocol.md` | Webcam data collection guide |

## License

License information will be added when the project license is finalized.
