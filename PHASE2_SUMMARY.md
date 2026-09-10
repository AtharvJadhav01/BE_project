# Phase 2: Facial Emotion/Expression Classifier Training Pipeline - COMPLETE

## Summary

Successfully built a complete training pipeline for facial emotion classification using MediaPipe features extracted from the FER2013 dataset.

## Dataset

- **Dataset**: FER2013 (CC BY 4.0 license)
- **Source**: https://www.kaggle.com/datasets/msambare/fer2013
- **Format**: 48x48 grayscale images organized by class folders (train/test)
- **Classes**: 7 emotions (angry, disgust, fear, happy, sad, surprised, neutral)
- **Samples**: 2,711 total (1,400 train, 1,311 test) with 200 samples/class max
- **Face Detection Rate**: 88.6% (2,401/2,711 images had detectable faces)

## Feature Extraction

- **Method**: MediaPipe FaceLandmarker → 468 landmarks + 52 blendshapes → 27 facial features
- **Features**: Eye openness, mouth parameters, eyebrow positions, head pose, symmetry, expressivity, nose sneer
- **Pipeline**: Reuses exact same feature extraction logic as live application
- **No-face handling**: Samples with no detected face are excluded (11.4% of dataset)

## Dataset Split

- **Train**: 1,054 samples (after 15% val split from training)
- **Validation**: 185 samples (split from training)
- **Test**: 1,162 samples (PublicTest set)

## Models Trained

### Random Forest (Best)
- **Parameters**: 200 estimators, max_depth=15, class_weight=balanced
- **Cross-validation F1 (weighted)**: 0.394 ± 0.008
- **Test Accuracy**: 43.0%
- **Test F1 (weighted)**: 0.408
- **Best per-class**: Surprised (F1=0.63), Happy (F1=0.58), Disgust (F1=0.42)
- **Worst per-class**: Fear (F1=0.19), Angry (F1=0.25)

### SVM (Baseline)
- **Test Accuracy**: 39.8%
- **Test F1 (weighted)**: 0.341
- **Worse than RF** on this feature set

## Files Created

```
training/
├── __init__.py
├── README.md                          # Dataset documentation
├── feature_extractor.py               # MediaPipe feature extraction (reused from app)
├── convert_to_csv_fast.py             # Image folder → CSV converter
├── dataset_loader.py                  # FER2013 loading + feature extraction
├── train_classifier.py                # Training script (RF/SVM/LR)
├── fer2013_features/                  # Extracted features (numpy arrays)
│   ├── X_train.npy, X_val.npy, X_test.npy
│   ├── y_train.npy, y_val.npy, y_test.npy
│   ├── scaler.joblib
│   └── metadata.json
├── models/
│   ├── rf_emotion_200est_15depth.joblib
│   ├── rf_emotion_200est_15depth_metadata.json
│   ├── svm_emotion_C10.0.joblib
│   └── svm_emotion_C10.0_metadata.json
└── convert_to_csv.py                  # Original converter
```

## Test Results

- **All 242 unit tests pass**
- **Mypy**: Clean (only numpy type stub warning, not our code)

## Performance Assessment

**The classifier is NOT production-ready** for 7-class emotion classification:
- ~43% accuracy is well below typical requirements (>70%)
- Fear and Angry classes poorly distinguished
- Features from MediaPipe blendshapes/landmarks insufficient for 7-class discrimination

## Recommendations for Phase 3+

1. **Better features**: Add geometric features (AU intensities, landmark distances)
2. **Better dataset**: Use RAF-DB or AffectNet (larger, better quality)
3. **Better model**: Fine-tune a lightweight CNN on raw crops, or use ensemble
4. **Fewer classes**: Merge similar emotions (fear+surprised, angry+disgust)
5. **Temporal smoothing**: Use temporal windows in live application

## Privacy Compliance

✅ No raw user images stored - only extracted 27-dim features
✅ FER2013 is public research dataset (CC BY 4.0)
✅ No live session data used for training
✅ Model only stores feature weights, no images