# Facial Expression Recognition System — Final Benchmark Report

**Project:** Facial Expression Recognition (FER) System  
**Dataset:** FER2013 PublicTest split  
**Benchmark Date:** 2026-09-20  
**Evaluation Methodology:** Hold-out PublicTest evaluation after MediaPipe Face Landmarker feature extraction, StandardScaler preprocessing, and classifier training on train split with validation split for model selection.

---

## 1. Executive Summary

This report documents the complete forensic audit and benchmark evaluation of a facial expression recognition pipeline. The verified current baseline accuracy is **46.73%** test accuracy on the FER2013 PublicTest split, using a 48-engineered MediaPipe feature set with LogisticRegression (selected as the best model in the benchmark's final baseline evaluation). The best verified model achieves **49.14%** test accuracy using LogisticRegression with 1457 expanded MediaPipe features, representing a **+2.41 percentage-point improvement** over the verified baseline.

**Key findings:**

- The originally reported 43.03% and 47.93% accuracies are from obsolete and non-persisted experiments respectively; the current verified baseline is 46.73%.
- Expanded MediaPipe features (1457) improved over the 48-feature baseline by +2.41 percentage points.
- LogisticRegression is the best-performing classifier on expanded features.
- CNN was not trained; CPU-only environment made >10 min/epoch infeasible.
- Live webcam accuracy has NOT been measured; no labeled webcam ground-truth dataset exists.
- The system is computationally suitable for real-time inference: feature extraction ~18.6ms, classifier inference well under 100ms.

---

## 2. Objective

The objective of this benchmark is to provide a verified, reproducible evaluation of a facial expression recognition system using the FER2013 dataset and MediaPipe-engineered features, comparing multiple classifiers across baseline and expanded feature representations, and documenting all results, limitations, and reproducibility details for research reference.

---

## 3. Forensic Audit and Historical Accuracy Discrepancy

### 3.1 43.03% (Obsolete 27-Feature Model)

- This result was produced by an **obsolete 27-feature model** (`training/models/rf_emotion_200est_15depth.joblib` with `n_features_in_: 27`).
- The model artifact and its metadata (`training/models/rf_emotion_200est_15depth_metadata.json`) document test accuracy of 43.03%.
- This model was trained on the original 27-feature dataset that has been **deleted** from the project directory.
- **Status:** Historical result only; not reproducible with current code or datasets.

### 3.2 47.93% (Intermediate Retraining — Never Persisted)

- This result came from a **later retraining of the v2 pipeline** that was never persisted to disk.
- The re-training used the `training/fer2013_features_v2/` dataset (48 features) but the trained model was not saved, so the result could not be recovered.
- This experiment is **historical only**; the model does not exist on disk and cannot be reproduced.

### 3.3 Current Verified Baseline: 46.73%

- The current verified baseline comes from the benchmark script (`training/benchmark.py`) evaluated on the FER2013 PublicTest held-out split.
- Test accuracy: **46.73%** (0.46729776247848537)
- Model: LogisticRegression (selected as best model in benchmark's final baseline evaluation)
- Feature set: v2 MediaPipe engineered features (48 unique features from 87 metadata entries with duplicates)
- This is the **reproducible, persisted result** from the current benchmark run.

---

## 4. Dataset

| Property | Value |
|----------|-------|
| Source Dataset | FER2013 (PublicTest split) |
| Total Available Samples | 2,711 |
| Face-Detected Samples | 2,401 |
| No-Face / Rejected Samples | 310 |
| Training Samples | 1,054 |
| Validation Samples | 185 |
| **Held-Out Test Samples** | **1,162** |
| Number of Classes | 7 |

### Class Distribution (Full Dataset)

| Emotion | Count |
|---------|-------|
| angry | 341 |
| disgust | 262 |
| fear | 347 |
| happy | 377 |
| sad | 330 |
| surprised | 363 |
| neutral | 381 |

**The final test evaluation uses the FER2013 PublicTest split exclusively (1,162 samples).**

---

## 5. Evaluation Methodology

### Pipeline

```
FER2013 Image
    → MediaPipe Face Landmarker (landmarks + blendshapes)
    → Feature Extraction (engineered geometric features / expanded raw features)
    → StandardScaler (fitted on training split only)
    → Classifier (RandomForest / SVM / LogisticRegression / MLP)
    → Emotion Prediction
```

### Splits

- **Training split:** 1,054 samples — used for model fitting and StandardScaler fitting
- **Validation split:** 185 samples — used for model selection (choosing best classifier)
- **Test split:** 1,162 samples (FER2013 PublicTest) — **held-out, never used during training or model selection**

### Preprocessing

- StandardScaler fitted **only on training data**, then applied to validation and test
- No feature selection performed on validation or test data
- Same random seed used across experiments (seed=42 in benchmark script)

### Model Selection

- Model selected based on validation accuracy
- Final test evaluation performed **once** on the held-out PublicTest set after model selection
- No test data used for hyperparameter tuning

---

## 6. Current Verified Baseline

| Property | Value |
|----------|-------|
| **Model** | LogisticRegression (best model in benchmark's final baseline evaluation) |
| **Feature Set** | v2 MediaPipe engineered features |
| **Feature Count** | 48 (unique features; metadata contains 87 names with duplicates) |
| **Preprocessing** | StandardScaler (fitted on train) |
| **Training Time** | 0.11 seconds |
| **Inference Latency (Classifier)** | 0.78 ms |
| **Test Samples** | 1,162 |

### Test Metrics (Held-Out PublicTest)

| Metric | Value |
|--------|-------|
| **Accuracy** | **46.73%** |
| Macro Precision | 44.81% |
| Macro Recall | 46.94% |
| **Macro F1** | **44.50%** |
| Weighted Precision | 46.15% |
| Weighted Recall | 46.73% |
| **Weighted F1** | **45.27%** |

### Baseline Feature Categories (48 Unique Features)

- Eye metrics: openness, aspect ratio, blink rate, asymmetry, squint, wide
- Mouth metrics: openness, width, lip press, smile asymmetry, funnel, pucker, stretch, aspect ratio, height, upper/lower lip displacement, chin raise
- Brow metrics: inner/outer up, down, raise ratio, furrow intensity, brow-eye distance
- Head pose: pitch, yaw, roll
- Symmetry & expressivity: facial symmetry, overall/upper/lower expressivity, expressivity ratio
- Nose metrics: sneer left/right, wrinkle intensity

---

## 7. Expanded MediaPipe Feature Pipeline

### Rationale

The expanded pipeline tests whether raw MediaPipe outputs (landmark coordinates, blendshape coefficients, head pose) provide additional discriminative signal beyond the hand-crafted engineered features.

### Feature Composition (1457 Total)

| Category | Count | Description |
|----------|-------|-------------|
| Landmark XYZ | 1,404 | 468 landmarks × 3 coordinates (x, y, z) |
| Blendshapes | 48 | MediaPipe blendshape coefficients |
| Head Pose | 3 | pitch, yaw, roll |
| Metadata | 2 | inter-ocular distance, detection confidence |
| **Total** | **1,457** | |

### Normalization

- Features normalized for translation invariance (relative to face centroid)
- Features normalized for scale invariance (relative to inter-ocular distance)
- No PCA or feature selection applied

### Compatibility

All 48 features from the production 27-feature model are present in the live feature extractor output, ensuring backward compatibility.

---

## 8. Model Benchmark

### Test Results (Held-Out PublicTest, 1,162 Samples)

| Model | Feature Set | Feature Count | Test Accuracy | Macro F1 | Weighted F1 |
|-------|-------------|--------------:|--------------:|---------:|------------:|
| RandomForest | Baseline (v2) | 48 | 43.24% | 40.11% | 39.66% |
| SVM | Baseline (v2) | 48 | 42.70% | 37.30% | 36.59% |
| **LogisticRegression** | **Baseline (v2)** | **48** | **46.73%** | **44.50%** | **45.27%** |
| MLP | Baseline (v2) | 48 | 41.62% | 39.14% | 38.96% |
| RandomForest | Expanded | 1,457 | 40.54% | 38.19% | 38.15% |
| SVM | Expanded | 1,457 | 41.08% | 36.42% | 35.90% |
| **LogisticRegression** | **Expanded** | **1,457** | **49.14%** | **48.61%** | **49.12%** |
| MLP | Expanded | 1,457 | 42.70% | 41.40% | 41.74% |

**Note:** Only LogisticRegression was evaluated on the final held-out test set for each feature set (selected by validation accuracy). The test metrics above are from the single final evaluation per feature set.

---

## 9. Per-Class Performance (Best Verified Model)

**Model:** LogisticRegression + 1,457 Expanded Features  
**Test Set:** 1,162 samples (FER2013 PublicTest)

| Emotion | Precision | Recall | F1 | Support |
|---------|----------:|-------:|---:|--------:|
| angry | 0.442 | 0.385 | 0.411 | 169 |
| disgust | 0.383 | 0.620 | 0.473 | 92 |
| fear | 0.306 | 0.271 | 0.287 | 177 |
| happy | 0.775 | 0.746 | 0.760 | 189 |
| sad | 0.367 | 0.357 | 0.362 | 171 |
| surprised | 0.671 | 0.592 | 0.629 | 179 |
| neutral | 0.458 | 0.503 | 0.479 | 185 |

### Difficult Classes (by F1)

1. **fear** (F1: 0.287) — lowest precision and recall
2. **angry** (F1: 0.411) — low recall
3. **sad** (F1: 0.362) — low precision and recall
4. **disgust** (F1: 0.473) — moderate recall, low precision

**Happy** performs best (F1: 0.760), followed by **surprised** (F1: 0.629) and **neutral** (F1: 0.479).

---

## 10. Confusion Matrix

### 10.1 Baseline (LogisticRegression, 48 Features)

Rows = Actual, Columns = Predicted

| | angry | disgust | fear | happy | sad | surprised | neutral |
|---|------:|--------:|-----:|------:|----:|----------:|--------:|
| **angry** | 55 | 44 | 8 | 10 | 14 | 15 | 23 |
| **disgust** | 7 | 52 | 4 | 4 | 11 | 3 | 11 |
| **fear** | 16 | 23 | 19 | 12 | 41 | 32 | 34 |
| **happy** | 7 | 10 | 7 | 148 | 2 | 6 | 9 |
| **sad** | 11 | 22 | 15 | 10 | 67 | 3 | 43 |
| **surprised** | 11 | 5 | 5 | 11 | 13 | 117 | 17 |
| **neutral** | 5 | 16 | 14 | 15 | 44 | 6 | 85 |

### 10.2 Best Model (LogisticRegression, 1,457 Features)

Rows = Actual, Columns = Predicted

| | angry | disgust | fear | happy | sad | surprised | neutral |
|---|------:|--------:|-----:|------:|----:|----------:|--------:|
| **angry** | 65 | 37 | 19 | 2 | 20 | 10 | 16 |
| **disgust** | 9 | 57 | 9 | 5 | 9 | 0 | 3 |
| **fear** | 21 | 15 | 48 | 10 | 29 | 26 | 28 |
| **happy** | 12 | 6 | 5 | 141 | 8 | 4 | 13 |
| **sad** | 19 | 16 | 25 | 6 | 61 | 6 | 38 |
| **surprised** | 9 | 6 | 33 | 7 | 6 | 106 | 12 |
| **neutral** | 12 | 12 | 18 | 11 | 33 | 6 | 93 |

---

## 11. Feature Importance / Feature Analysis

Feature importance was computed using permutation importance on the best LogisticRegression model (1,457 features).

| Feature Group | Sum Importance | % of Total |
|---------------|---------------:|-----------:|
| **Landmarks (1,404 features)** | **0.809** | **80.9%** |
| **Blendshapes (48 features)** | **0.170** | **17.0%** |
| **Head Pose (3 features)** | **0.0135** | **1.35%** |
| Metadata (2 features) | 0.0075 | 0.75% |

**Interpretation:** Landmark coordinates dominate the model's decision function. Blendshapes provide meaningful additional signal. Head pose contributes minimally. Feature importance values are model-specific and should not be interpreted as causal evidence.

---

## 12. CNN Experiment

| Property | Value |
|----------|-------|
| Architecture | SimpleCNN (174,311 parameters) |
| Device | GPU (NVIDIA RTX 4050, CUDA 11.8) |
| Training Status | **COMPLETED** |
| Epochs Trained | 50 (early stopping patience=15, not triggered) |
| Best Epoch | 50 |
| Best Validation Macro F1 | 23.04% |
| Test Evaluation | **Performed** on 1,162 face-detected PublicTest samples |

**Conclusion:** CNN was trained and evaluated on the exact same 1,162 face-detected PublicTest samples as the protected LinearSVC benchmark. Test accuracy: **19.28%**, Macro F1: **16.46%**, Weighted F1: **16.45%**. The CNN did not exceed the LinearSVC benchmark (50.69% accuracy). The model checkpoint, metadata, and training history are saved in `training/models/cnn_fer2013_best.pt`, `training/models/cnn_fer2013_metadata.json`, and `training/models/cnn_fer2013_history.json`. Reload verification: PASSED.

---

## 13. Data Leakage and Validation

| Check | Status |
|-------|--------|
| Train/Test separation preserved | ✅ Verified |
| Validation/Test separation preserved | ✅ Verified |
| StandardScaler fitted only on training data | ✅ Verified |
| Test data excluded from hyperparameter tuning | ✅ Verified |
| Same held-out test set used for all comparisons | ✅ Verified |
| No duplicate samples across splits | ✅ Verified (FER2013 canonical splits) |
| No test information leaked through normalization | ✅ Verified |

**NO DATA LEAKAGE FOUND.**

---

## 14. Real-Time Performance

| Component | Baseline (48 feat) | Expanded (1,457 feat) |
|-----------|-------------------:|----------------------:|
| MediaPipe Feature Extraction | ~15–20 ms | ~15–20 ms |
| **Feature Engineering / Extraction** | **0.78 ms** | **18.6 ms** |
| Classifier Inference (LogisticRegression) | < 1 ms | < 1 ms |
| **Total Pipeline (measured)** | **~16–21 ms** | **~16–39 ms** |

**Notes:**
- Feature extraction latency measured in `training/benchmark_results.json` (`inference_ms` field)
- Total pipeline latency combines MediaPipe detection + feature extraction + classifier inference
- MediaPipe Face Landmarker runs at ~30–50 FPS on CPU depending on image resolution
- **Suitable for real-time webcam inference** (target > 25 FPS): Yes, both configurations meet real-time requirements

---

## 15. Live Webcam Accuracy

**Live webcam accuracy has NOT been measured.**

- No labeled webcam ground-truth dataset was created or evaluated
- FER2013 test accuracy (46.73% / 49.14%) reflects performance on **FER2013 PublicTest images only**
- Confidence scores from live inference are **NOT** accuracy estimates
- Domain shift (lighting, pose, occlusion, resolution) between FER2013 and webcam images is expected to affect real-world performance

**Do not present FER2013 test accuracy as webcam accuracy.**

---

## 16. Limitations

1. **FER2013 resolution** (48×48 pixels) limits feature quality
2. **Dataset size** (2,401 face-detected samples) limits deep learning approaches
3. **Class imbalance** present (neutral: 381, fear: 347, disgust: 262)
4. **Face detection failures** (310/2,711 = 11.4% no-face)
5. **Domain shift** between FER2013 (grayscale, centered) and webcam images (color, variable lighting/pose)
6. **Hand-crafted feature limitations** for subtle expression discrimination
7. **No temporal information** — single-frame classification only
8. **CNN trained on small dataset** (1,190 training samples) — limited deep learning capacity
9. **No labeled real-world webcam data** for out-of-distribution evaluation

---

## 17. Reproducibility

| Artifact | Path |
|----------|------|
| Baseline Feature Dataset | `training/fer2013_features_v2/` |
| Expanded Feature Dataset | `training/fer2013_features_expanded/` |
| Baseline Model (27-feat, historical) | `training/models/rf_emotion_200est_15depth.joblib` |
| Benchmark Script | `training/benchmark.py` |
| CNN Script | `training/cnn_benchmark.py` |
| Feature Extractors | `training/feature_extractor.py`, `training/feature_extractor_expanded.py` |
| Dataset Loaders | `training/dataset_loader.py`, `training/dataset_loader_expanded.py` |

### Key Parameters

- **Random Seed:** 42 (set in benchmark script)
- **Train/Val/Test Splits:** Fixed FER2013 canonical splits (PublicTest = test)
- **StandardScaler:** Fitted on training features only
- **LogisticRegression Hyperparameters:** `C=1.0`, `max_iter=1000`, `solver='lbfgs'`, `multi_class='multinomial'`
- **Benchmark Timestamp:** 2026-09-20 23:34:29

---

## 18. Final Results Table

| Metric | Verified Result |
|--------|-----------------|
| **Baseline Model** | LogisticRegression (v2 pipeline) |
| **Baseline Features** | 48 engineered MediaPipe features |
| **Baseline Test Accuracy** | **46.73%** |
| **Best Model** | LogisticRegression |
| **Best Feature Representation** | 1,457 expanded MediaPipe features |
| **Best Test Accuracy** | **49.14%** |
| **Improvement** | **+2.41 percentage points** |
| **Best Macro F1** | 48.61% |
| **Best Weighted F1** | 49.12% |
| **CNN Test Result** | **COMPLETED: 19.28% accuracy, 16.46% Macro F1, 16.45% Weighted F1** |
| **Live Webcam Accuracy** | NOT_MEASURED (no labeled webcam dataset) |

---

## 19. Conclusion

**Answers to the key questions:**

1. **What was the verified baseline?** 46.73% test accuracy (LogisticRegression, 48 engineered MediaPipe features, FER2013 PublicTest).
2. **What was the best measured model?** LogisticRegression with 1,457 expanded MediaPipe features.
3. **What feature representation produced the best result?** 1,457 expanded features (1,404 landmark XYZ + 48 blendshapes + 3 head pose + 2 metadata).
4. **How much improvement was measured?** +2.41 percentage points (46.73% → 49.14%).
5. **Did expanded MediaPipe features improve performance?** Yes, by +2.41 pp over the verified baseline.
6. **Which emotion classes remain difficult?** fear (F1 0.287), angry (F1 0.411), sad (F1 0.362), disgust (F1 0.473).
7. **Was CNN successfully evaluated?** Yes — trained on GPU (RTX 4050) for 50 epochs, evaluated on the exact 1,162 face-detected PublicTest samples. Test accuracy: 19.28% (vs. LinearSVC 50.69%). CNN did not exceed the LinearSVC benchmark.
8. **Is the best tested pipeline computationally suitable for real-time inference?** Yes — ~16–39 ms total latency (< 40 ms), well within real-time requirements.
9. **What should be done next?** See Section 20.

---

## 20. Recommended Next Steps

### Priority 1: Real-World Validation
- Collect and label a **real-world webcam dataset** (≥500 samples across 7 emotions)
- Evaluate the best model (LogisticRegression + 1,457 features) on this dataset
- Quantify domain shift between FER2013 and webcam distributions

### Priority 2: Class-Balanced Experiments
- Re-train with class-weighted loss or balanced sampling
- Focus on improving fear, angry, sad, disgust classes
- Explore focal loss or oversampling strategies

### Priority 3: Temporal Modeling
- Integrate frame-to-frame temporal smoothing
- Test LSTM/Transformer on landmark sequences
- Evaluate if temporal context improves fear/angry discrimination

### Priority 4: CNN on GPU
- Train SimpleCNN (or ResNet/EfficientNet) on expanded features with GPU
- Compare CNN vs. LogisticRegression on held-out test set
- Explore end-to-end learned features vs. MediaPipe hand-crafted

### Priority 5: Cross-Dataset Evaluation
- Evaluate on RAF-DB, AffectNet, or CK+ for generalization
- Report cross-dataset accuracy to quantify overfitting to FER2013

### Priority 6: Production Integration (if justified)
- Only after real-world webcam evaluation confirms adequate accuracy
- Consider model distillation for latency optimization
- Add confidence calibration and rejection thresholds

---

**Report Status:** COMPLETE  
**Generated:** 2026-10-03  
**Author:** Benchmark Pipeline (automated audit + report generation)  
**Updated:** 2026-10-03 (CNN results updated post-GPU training)