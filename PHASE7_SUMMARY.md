# Phase 7: Temporal Smoothing for Emotion Classifier

## Summary

Implemented temporal smoothing using the existing `EmotionSmoother` class (EMA-based) to reduce frame-to-frame expression flickering and provide stable emotion estimations in the live face-analysis pipeline.

## Changes Made

### 1. Fixed `src/analysis/face/state.py` - Added `get_logger` import
- Added `from core.logging import get_logger` to fix `NameError` when `EmotionSmoother` initializes its logger
- The `EmotionSmoother.__init__` calls `get_logger(__name__)` at line 428

### 2. Fixed `src/face/capture.py` - Fixed broken import line
- Fixed syntax error where `from analysis.face.state import EmotionSmoother` was concatenated with the previous import using literal `\n` instead of actual newline
- The import now properly spans two lines

### 3. Integration verified - No new code needed
- The `_run_face_analysis()` method in `capture.py` already integrates the `EmotionSmoother`:
  - Adds raw classifier predictions to the smoother (line 536)
  - Gets smoothed EMA result (line 539)
  - Creates `expression_estimate` using **smoothed** expression values (lines 542-548)
- The `_poll_face_analysis()` method in `main_window.py` reads the smoothed expression from `_last_analysis_result` and displays it (lines 989-1011)

### 4. Configuration added to `config/config.yaml` under `analysis.face`
- `emotion_confidence_threshold: 0.5` - minimum confidence threshold
- `emotion_smoothing_window: 5` - EMA window size
- `emotion_smoothing_alpha: 0.3` - EMA alpha factor

## Test Results

- **267/267 tests pass** (previously 4 failures due to `get_logger` import issue)
- All test categories pass: unit tests, integration tests, UI tests, face analysis tests
- No regressions introduced

## Architecture

The smoothing pipeline flows as:

1. `_run_face_analysis()` extracts features, predicts emotion class probabilities
2. Probabilities are fed to `EmotionSmoother.add_prediction()` 
3. EMA computes smoothed probabilities: `smoothed = alpha * raw + (1 - alpha) * previous`
4. Smoothed expression estimate is stored in `analysis_result["expression_estimate"]`
5. `_poll_face_analysis()` reads the smoothed estimate and updates the UI display

This ensures that expression labels change only when the smoothed confidence is sustained across multiple frames, eliminating jitter from individual frame predictions.

## Files Modified

- `src/analysis/face/state.py` - Added `get_logger` import
- `src/face/capture.py` - Fixed broken import line (import now correct)
- `config/config.yaml` - Added `emotion_confidence_threshold`, `emotion_smoothing_window`, `emotion_smoothing_alpha`

## No Changes Needed

- `src/ui/main_window.py` - Already reads smoothed expression from `_last_analysis_result`
- `src/face/capture.py:_run_face_analysis()` - Already uses smoothed expressions
- Feature extraction (37-dimensional representation) - Preserved unchanged
- Model retraining - Not performed (uses existing Random Forest model)