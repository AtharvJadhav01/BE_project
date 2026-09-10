# Phase 3 Implementation Plan: Improve Facial Emotion Classifier

## Current Baseline (Phase 2)
- **Model**: Random Forest (200 trees, max_depth=15)
- **Features**: 27 (eye, mouth, eyebrow, head pose, movement, symmetry, expressivity)
- **Test Accuracy**: 43.0%
- **Weighted F1**: 0.408
- **Macro F1**: ~0.38

## Confusion Matrix Analysis

| True\Pred | Angry | Disgust | Fear | Happy | Sad | Surprised | Neutral |
|-----------|-------|---------|------|-------|-----|-----------|---------|
| **Angry**   | 31    | 39      | 9    | 20    | 28  | 19        | 23      |
| **Disgust** | 5     | 51      | 3    | 8     | 14  | 5         | 6       |
| **Fear**    | 9     | 16      | 24   | 25    | 33  | 38        | 32      |
| **Happy**   | 10    | 15      | 5    | 114   | 6   | 13        | 26      |
| **Sad**     | 11    | 16      | 15   | 10    | 63  | 13        | 43      |
| **Surprised**| 6    | 5       | 5    | 11    | 10  | 133       | 9       |
| **Neutral** | 5     | 11      | 13   | 13    | 37  | 22        | 84      |

**Per-class Recall**: Angry=18%, Disgust=55%, Fear=14%, Happy=60%, Sad=37%, Surprised=74%, Neutral=45%

## Key Confusion Patterns

1. **Angry (18% recall)**: Confused with Neutral(23), Sad(28), Happy(20), Disgust(39)
2. **Fear (14%)**: Confused with Sad(33), Surprised(38), Neutral(32), Happy(25)
3. **Sad (37%)**: Confused with Neutral(43), Fear(33), Angry(11)
4. **Neutral (45%)**: Confused with Sad(37), Fear(13), Surprised(22)
4. **Disgust** ↔ **Angry** confusion (39→1)

## New Features to Add (8 features)

| Feature | Source | Target Confusion | Formula |
|---------|--------|------------------|---------|
| `eye_white_exposure_ratio` | Landmarks (iris + eyelid) | Fear vs Surprise | Visible sclera / eye height |
| `mouth_corner_depression` | Landmarks (corners vs center) | Sad vs Neutral | (center_y - corner_y) / face_height |
| `brow_furrow_depth` | Landmarks (brow inner distance) | Angry vs Disgust | Inner brow distance |
| `eye_white_exposure_ratio` | Iris + eyelid landmarks | Fear vs Surprise | (sclera_visible / eye_height) |
| `mouth_corner_depression` | Landmarks (corners vs center) | Sad vs Neutral | (center_y - corner_y) / face_height |
| `upper_lid_raise` | Blendshape (eyeWide) + landmarks | Fear vs Surprise | eyeWide blendshape |
| `lower_lip_depressor` | Blendshape (mouthLowerDownL/R) | Sad | mouthLowerDownL + mouthLowerDownRight |
| `chin_boss_activation` | Blendshape (mouthShrugLower/Upper) | Sad | mouthShrugLower + mouthShrugUpper |
| `nostril_flare` | Blendshape (noseSneer) | Disgust vs Angry | noseSneerLeft + noseSneerRight |

## Files to Modify

1. `training/feature_extractor.py` - Add new feature extraction methods
2. `src/analysis/face/features.py` - Keep in sync (application-side extractor)
3. `src/analysis/face/features.py` - Update FacialFeatures dataclass & get_feature_names()
4. `training/feature_extractor.py` - Add new feature extraction methods
5. `training/dataset_loader.py` - Regenerate dataset with new features
6. `training/train_classifier.py` - Retrain models
7. `tests/unit/test_face_analysis.py` - Add tests for new features

## Implementation Steps

1. **Add new features to FacialFeatures dataclass** (both files)
2. **Add extraction methods** to feature extractors (both files)
3. **Update get_feature_names()** in both extractors
3. **Update _features_to_dict()** to include new features
4. **Regenerate dataset** with new features (training/dataset_loader.py)
4. **Retrain models** (RF, SVM, LR) and evaluate
5. **Run tests** - ensure all pass

## Evaluation Criteria

- **Primary**: Test accuracy > 50% (meaningful improvement over 43%)
- **Secondary**: Macro F1 > 0.45, Weighted F1 > 0.45
- **Per-class**: Fear recall > 25%, Angry recall > 30%, Sad recall > 50%
- No test set tuning

## Files to Modify

1. `src/analysis/face/features.py` - Application feature extractor
2. `training/feature_extractor.py` - Training feature extractor
3. `training/dataset_loader.py` - Regenerate dataset
3. `training/train_classifier.py` - Retrain and evaluate
4. `tests/unit/test_face_analysis.py` - Add tests for new features