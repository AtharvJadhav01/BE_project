# Real-World Facial Expression Validation Report

## Status

**REAL-WORLD ACCURACY: MEASURED (2026-10-03)**

A labeled webcam dataset has been collected and evaluated against the verified LinearSVC model.

**Verified Webcam Evaluation Results:**
- **Accuracy: 18.79%**
- **Macro F1: 8.62%**
- **Weighted F1: 9.69%**

**Model Evaluated:**
- LinearSVC (1,449 features: 1,404 landmarks + 48 blendshapes)
- Artifacts: `training/models/linear_svc_optimized_1449.joblib` + `scaler_optimized_1449.joblib`
- FER2013 PublicTest benchmark: 50.69% accuracy, 49.19% Macro F1, 49.98% Weighted F1

---

## Validation Methodology

### Model Being Evaluated
- **Model**: LinearSVC (current best, 50.69% FER2013 accuracy)
- **Features**: 1,449 MediaPipe landmarks + blendshapes
- **Training Data**: FER2013 (1,239 face-detected samples: 1,054 train + 185 val)
- **FER2013 PublicTest Benchmark**: 50.69% accuracy, 49.19% Macro F1, 49.98% Weighted F1

### Evaluation Approach
1. Collect labeled webcam dataset using `tools/webcam_dataset_collector.py`
2. Validate dataset using `tools/validate_webcam_dataset.py`
3. Evaluate using `training/evaluate_webcam.py` (NO retraining)
4. Compare webcam performance vs FER2013 benchmark

### Domain Shift Measurement
This evaluation measures the domain shift between:
- **FER2013**: Grayscale, 48×48px, centered faces, constrained poses
- **Webcam**: Color, variable resolution, natural head poses, lighting variation

### Data Splits
- **No retraining**: Webcam dataset used ONLY as test set
- **Split by session**: If single participant, separate recording sessions
- **Split by participant**: If multiple participants, separate by person
- **Never mix**: Frames from same session never in both train and test

---

## Dataset Requirements

| Requirement | Specification |
|-------------|---------------|
| Classes | 7 (angry, disgust, fear, happy, sad, surprised, neutral) |
| Min samples per class | 30 |
| Recommended per class | 50-80 |
| Total minimum | 250 samples |
| Total recommended | 400+ samples |
| Session separation | Yes (no frame-level mixing) |
| Label source | Manual selection during recording |
| Metadata | JSONL with sample_id, class, timestamp, session_id, image_path |

---

## Evaluation Metrics

The evaluation computes:

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Macro Precision | Per-class precision averaged |
| Macro Recall | Per-class recall averaged |
| **Macro F1** | **Primary metric (class-balanced)** |
| Weighted Precision | Support-weighted precision |
| Weighted Recall | Support-weighted recall |
| **Weighted F1** | **Support-weighted F1** |
| Per-class P/R/F1 | Identify weak classes |
| Confusion Matrix | Error pattern analysis |

---

## Current Status

| Component | Status |
|-----------|--------|
| Webcam dataset collector | ✅ Created (`tools/webcam_dataset_collector.py`) |
| Dataset validator | ✅ Created (`tools/validate_webcam_dataset.py`) |
| Evaluation tool | ✅ Created (`training/evaluate_webcam.py`) |
| Protocol documentation | ✅ Created (`docs/webcam_dataset_protocol.md`) |
| Webcam dataset | ✅ COLLECTED (1,804 images, 7 classes) |
| Real-world evaluation | ✅ COMPLETED (2026-10-03) |
| Best model artifact (LinearSVC 1449 feat) | ✅ SAVED (`training/models/linear_svc_optimized_1449.joblib`) |

### Evaluation Complete

**The LinearSVC model (1,449 features) was evaluated on the webcam dataset.**

The `training/models/` directory contains:
- `linear_svc_optimized_1449.joblib` (50.69% test accuracy)
- `scaler_optimized_1449.joblib`
- `linear_svc_optimized_1449_metadata.json`

**No retraining needed. Evaluation complete.**

---

## Verified Results (2026-10-03)

### Comparison Table

| Dataset | Model | Accuracy | Macro F1 | Weighted F1 |
|---------|-------|----------|----------|-------------|
| FER2013 PublicTest | LinearSVC (1449 feat) | 50.69% | 49.19% | 49.98% |
| **Webcam Test Set** | **Same LinearSVC** | **18.79%** | **8.62%** | **9.69%** |

### Per-Class Webcam Performance

| Expression | Precision | Recall | F1 | Support |
|------------|-----------|--------|----|---------|
| angry | 0.000 | 0.000 | 0.000 | 326 |
| disgust | 0.000 | 0.000 | 0.000 | 178 |
| fear | 0.138 | 0.471 | 0.213 | 331 |
| happy | 0.000 | 0.000 | 0.000 | 234 |
| sad | 0.000 | 0.000 | 0.000 | 249 |
| surprised | 0.273 | 0.685 | 0.391 | 267 |
| neutral | 0.000 | 0.000 | 0.000 | 219 |

### Domain Shift Analysis

| Metric | FER2013 PublicTest | Webcam Test Set | Gap |
|--------|-------------------|-----------------|-----|
| Accuracy | 50.69% | 18.79% | **−31.90 pp** |
| Macro F1 | 49.19% | 8.62% | **−40.57 pp** |
| Weighted F1 | 49.98% | 9.69% | **−40.29 pp** |

A substantial domain shift was observed. Performance on the held-out webcam dataset was substantially lower than on FER2013. This demonstrates that benchmark performance does not directly translate to real-world webcam performance.

The webcam dataset contains 1,804 manually labeled images across 7 classes (angry: 326, disgust: 178, fear: 331, happy: 234, sad: 249, surprised: 267, neutral: 219). The webcam dataset is not part of FER2013 training and is used as an external/held-out real-world evaluation set. The webcam result should NOT be presented as the FER2013 benchmark accuracy. Confidence scores must not be interpreted as accuracy.

### Error Analysis

| Error Type | Count | Examples |
|------------|-------|----------|
| Head pose extremes | TBD | |
| Lighting variation | TBD | |
| Distance variation | TBD | |
| Occlusion | TBD | |
| Low detection confidence | TBD | |
| Expression ambiguity | TBD | |

**Domain shift factors (plausible contributors, not proven):** lighting variation, pose variation, resolution differences, camera characteristics, subject variation, and dataset differences may contribute to the gap.

---

## Next Actions

The webcam evaluation is complete. Next steps:

1. **Analyze error patterns** in detail (per-class performance, confusion matrix)
2. **Cross-dataset evaluation** (RAF-DB, AffectNet, CK+) to quantify generalization
3. **Temporal modeling** (LSTM/Transformer on landmark sequences) to improve difficult classes
4. **Class-balanced training** for fear/sad/angry improvement
5. **Confidence calibration** for reliable probability estimates
6. **Cross-dataset evaluation** (RAF-DB, AffectNet, CK+) for generalization

```bash
# Evaluation already complete - results in:
# data/webcam_expression_dataset/evaluation_results.json
```

---

## Privacy & Ethics

- Dataset stored locally only
- No personal identifiers in metadata
- Session IDs are randomized
- Raw video not stored (only sampled frames)
- Participant consent documented separately if applicable
- Data deletion policy: Delete after evaluation unless retained for retraining

---

*Report generated as template. Update after actual evaluation.*