# Real-World Facial Expression Validation Report

## Status

**REAL-WORLD ACCURACY: NOT YET MEASURED**

No labeled webcam dataset has been collected yet.

---

## Validation Methodology

### Model Being Evaluated
- **Model**: LogisticRegression (current best)
- **Features**: 1,457 expanded MediaPipe features
  - 1,404 landmark coordinates (468 × 3, normalized)
  - 52 blendshape coefficients
  - 3 head pose (pitch, yaw, roll)
  - 2 metadata (inter-ocular distance, detection confidence)
- **Training Data**: FER2013 (2,401 face-detected samples)
- **FER2013 PublicTest Benchmark**: 49.14% accuracy, 48.61% Macro F1, 49.12% Weighted F1

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
| Webcam dataset | ❌ NOT COLLECTED |
| Real-world evaluation | ❌ NOT RUN |
| Best model artifact (1,457-feat LR) | ❌ NOT SAVED |

### Critical Blocker

**The 1,457-feature LogisticRegression model was never persisted to disk.**

The benchmark phase evaluated this model in-memory but did not save the artifact. The `training/models/` directory only contains:
- `rf_emotion_200est_15depth.joblib` (27 features, 43.03%)
- `svm_emotion_C10.0.joblib` (27 features)

**Action Required**: Retrain and save the LogisticRegression model on 1,457 expanded features before webcam evaluation can proceed.

---

## Expected Results Template

Once data is collected and model is available:

### Comparison Table

| Dataset | Model | Accuracy | Macro F1 | Weighted F1 |
|---------|-------|----------|----------|-------------|
| FER2013 PublicTest | LogisticRegression (1457 feat) | 49.14% | 48.61% | 49.12% |
| Webcam Test Set | Same model | TBD | TBD | TBD |

### Per-Class Webcam Performance

| Expression | Precision | Recall | F1 | Support |
|------------|-----------|--------|----|---------|
| angry | TBD | TBD | TBD | TBD |
| disgust | TBD | TBD | TBD | TBD |
| fear | TBD | TBD | TBD | TBD |
| happy | TBD | TBD | TBD | TBD |
| sad | TBD | TBD | TBD | TBD |
| surprised | TBD | TBD | TBD | TBD |
| neutral | TBD | TBD | TBD | TBD |

### Error Analysis (To Be Completed)

| Error Type | Count | Examples |
|------------|-------|----------|
| Head pose extremes | TBD | |
| Lighting variation | TBD | |
| Distance variation | TBD | |
| Occlusion | TBD | |
| Low detection confidence | TBD | |
| Expression ambiguity | TBD | |

---

## Next Actions

1. **Retrain and save 1,457-feature LogisticRegression model**
   ```bash
   # Run training script for expanded features
   python training/train_classifier.py --features expanded --model logistic_regression
   ```

2. **Collect webcam dataset**
   ```bash
   python tools/webcam_dataset_collector.py
   ```

3. **Validate dataset**
   ```bash
   python tools/validate_webcam_dataset.py
   ```

4. **Evaluate model**
   ```bash
   python training/evaluate_webcam.py
   ```

5. **Document results** in this file

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