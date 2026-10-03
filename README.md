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

All artifacts include metadata with hyperparameters, seeds, split info; reload verification passed for all.

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
