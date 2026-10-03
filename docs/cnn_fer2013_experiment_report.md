# CNN FER2013 Experiment Report

**Project:** Facial Expression Recognition (FER) System  
**Experiment:** CNN baseline on raw FER2013 48×48 grayscale images  
**Date:** 2026-10-03  
**GPU:** NVIDIA RTX 4050 (CUDA 11.8, PyTorch 2.7.1+cu118)

---

## 1. Objective

Train and evaluate a CNN directly on FER2013 raw pixel images using the exact same 1,162 face-detected PublicTest samples as the protected LinearSVC benchmark, to determine whether a CNN can exceed the LinearSVC test accuracy of 50.69%.

---

## 2. Dataset & Exact Test Protocol

### 2.1 Why 1,162 Instead of 1,311

The raw FER2013 CSV contains 1,311 PublicTest samples. However, the protected LinearSVC benchmark uses MediaPipe Face Landmarker features, which **failed to detect faces in 149 PublicTest samples (11.4%)**. The LinearSVC test set consists exclusively of the 1,162 samples where MediaPipe successfully detected a face.

For a fair comparison, the CNN **must** evaluate on the identical 1,162 samples. The mapping is saved in `training/cnn_benchmark_split.json` and was reconstructed by running MediaPipe face detection on all 1,311 PublicTest images.

### 2.2 Test Set Class Distribution (1,162 samples)

| Emotion | Count |
|---------|-------|
| angry   | 169   |
| disgust | 92    |
| fear    | 177   |
| happy   | 189   |
| sad     | 171   |
| surprised | 179 |
| neutral | 185   |

### 2.3 Training/Validation Split

- **Training pool:** `Usage == "Training"` (1,400 samples, 200 per class)
- **Train:** 1,190 samples (85%, stratified, `random_state=42`)
- **Validation:** 210 samples (15%, stratified)
- **Test:** 1,162 fixed samples (face-detected PublicTest)
- **No overlap** between any splits verified

---

## 3. Preprocessing & Augmentation

| Component | Setting |
|-----------|---------|
| Input normalization | Pixels / 255.0 → [0, 1] |
| Input shape | (1, 48, 48) grayscale |
| Training augmentation | Horizontal flip (50%), rotation ±10°, translation ±2px |
| Validation/Test augmentation | None |
| Class weights | Uniform [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] (training set is balanced) |

---

## 4. Architecture

**SimpleCNN** — parameter count: **174,311**

```
Input: 1×48×48
Conv1: 1→32, 3×3, BN, ReLU, 32→32, 3×3, BN, ReLU, MaxPool2×2, Dropout2d(0.25)  → 24×24
Conv2: 32→64, 3×3, BN, ReLU, 64→64, 3×3, BN, ReLU, MaxPool2×2, Dropout2d(0.25) → 12×12
Conv3: 64→128, 3×3, BN, ReLU, MaxPool2×2, Dropout2d(0.25)                        → 6×6
GlobalAvgPool → 128
Linear(128→256), ReLU, Dropout(0.3)
Linear(256→7)
```

**Weight initialization:** Conv2d — kaiming_normal (fan_out, ReLU); Linear — normal(0, 0.01)

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | Patience=15 on validation Macro F1 |
| Seeds | Python/NumPy/PyTorch/CUDA = 42 |
| Device | CUDA (RTX 4050) |

---

## 6. Training Summary

- **Training time:** 1,371 seconds (~23 minutes)
- **Best epoch:** 50 (no early stopping triggered)
- **Best validation Macro F1:** 0.2304
- **Final training loss:** 1.83
- **Final validation Macro F1:** 0.2304 (epoch 50)

Training curves showed steady improvement; early stopping did not trigger within 50 epochs.

---

## 7. Final Test Metrics (1,162 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | **19.28%** |
| **Macro F1** | **16.46%** |
| **Weighted F1** | **16.45%** |

### Per-Class Performance

| Emotion | Precision | Recall | F1 | Support |
|---------|----------:|-------:|---:|--------:|
| angry   | 0.116 | 0.030 | 0.047 | 169 |
| disgust | 0.107 | 0.641 | 0.184 | 92 |
| fear    | 0.000 | 0.000 | 0.000 | 177 |
| happy   | 0.187 | 0.090 | 0.121 | 189 |
| sad     | 0.289 | 0.088 | 0.135 | 171 |
| surprised | 0.444 | 0.469 | 0.457 | 179 |
| neutral | 0.186 | 0.238 | 0.209 | 185 |

### Confusion Matrix (rows=true, cols=pred)

| | angry | disgust | fear | happy | sad | surprised | neutral |
|---|------:|--------:|-----:|------:|----:|----------:|--------:|
| **angry**   | 5 | 68 | 0 | 22 | 9 | 27 | 38 |
| **disgust** | 3 | 59 | 0 | 7 | 1 | 12 | 10 |
| **fear**    | 5 | 81 | 0 | 11 | 9 | 33 | 38 |
| **happy**   | 9 | 111 | 0 | 17 | 1 | 13 | 38 |
| **sad**     | 5 | 84 | 0 | 10 | 15 | 11 | 46 |
| **surprised** | 8 | 53 | 0 | 4 | 8 | 84 | 22 |
| **neutral** | 8 | 95 | 0 | 20 | 9 | 9 | 44 |

**Failed classes:** fear (0% recall), angry (3% recall), happy (9% recall), sad (9% recall)  
**Best classes:** surprised (47% recall), disgust (64% recall), neutral (24% recall)

---

## 8. Comparison with Protected LinearSVC Benchmark

| Metric | LinearSVC (Protected) | CNN (This Experiment) | Difference |
|--------|---------------------:|---------------------:|-----------:|
| Accuracy | **50.69%** | 19.28% | **−31.41 pp** |
| Macro F1 | **49.19%** | 16.46% | **−32.73 pp** |
| Weighted F1 | **49.98%** | 16.45% | **−33.53 pp** |

The CNN did **not** exceed the LinearSVC benchmark on any metric. The LinearSVC remains the protected baseline.

---

## 9. Limitations

1. **Small training set** (1,190 samples) limits deep learning capacity
2. **FER2013 resolution** (48×48) provides limited visual information
3. **Grayscale only** — no color cues
4. **Class difficulty** — fear, angry, happy, sad remain near-zero recall
3. **No face alignment** — CNN receives raw crops; MediaPipe provides aligned landmarks
4. **No external data** — training from scratch on FER2013 only
4. **Single-frame only** — no temporal modeling

---

## 10. Reproducibility

| Artifact | Path |
|----------|------|
| Model checkpoint | `training/models/cnn_fer2013_best.pt` |
| Metadata | `training/models/cnn_fer2013_metadata.json` |
| Training history | `training/models/cnn_fer2013_history.json` |
| Test split mapping | `training/cnn_benchmark_split.json` |
| Training script | `training/train_cnn.py` |

**Reload verification:** ✅ PASSED — predictions and metrics identical after reloading checkpoint

**Seeds recorded:** Python=42, NumPy=42, PyTorch=42, CUDA=42

---

## 11. Conclusion

The CNN trained on raw FER2013 48×48 grayscale images achieved **19.28% test accuracy** on the exact 1,162 face-detected PublicTest samples used by the protected LinearSVC benchmark (50.69% accuracy). The CNN did not approach the LinearSVC performance (−31.41 percentage points on accuracy).

The primary limitation is the small training set (1,190 samples) combined with the low-resolution, grayscale FER2013 images. The LinearSVC benefits from MediaPipe's engineered facial landmarks (1,449 features) which provide strong geometric priors, while the CNN must learn facial structure from scratch on very limited data.

**Recommendation:** Do not deploy this CNN for expression recognition. The LinearSVC with MediaPipe features remains the recommended model for this task.

---

## 12. Artifacts Saved

- `training/models/cnn_fer2013_best.pt` — model checkpoint
- `training/models/cnn_fer2013_metadata.json` — full metadata
- `training/models/cnn_fer2013_history.json` — training history
- `docs/cnn_fer2013_experiment_report.md` — this report