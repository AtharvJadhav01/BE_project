# Facial Expression Classifier Training Pipeline

## Dataset: FER2013

**Source**: https://www.kaggle.com/datasets/msambare/fer2013

**License**: FER2013 is released under the **CC BY 4.0** license (Creative Commons Attribution 4.0 International). 

**Attribution**: "FER2013 dataset" by Pierre-Luc Carrier and Aaron Courville, used under CC BY 4.0.

**Dataset Statistics**:
- 35,887 grayscale 48x48 pixel images
- 7 emotion classes: Angry (0), Disgust (1), Fear (2), Happy (3), Sad (4), Surprise (5), Neutral (6)
- Train: 28,709 | Public Test: 3,589 | Private Test: 3,589

## Label Mapping

| FER2013 Label | Emotion | App Label |
|---------------|---------|-----------|
| 0 | Angry | angry |
| 1 | Disgust | disgust |
| 2 | Fear | fear |
| 3 | Happy | happy |
| 4 | Sad | sad |
| 5 | Surprise | surprised |
| 6 | Neutral | neutral |

## Pipeline Overview

1. **Feature Extraction**: Image → MediaPipe FaceLandmarker → 468 landmarks + 52 blendshapes → FacialFeatures (27-dim)
2. **Dataset Processing**: Extract features from all FER2013 images, handle no-face cases
3. **Training**: Random Forest / SVM on 27-dim feature vectors
3. **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, Per-class metrics
4. **Model Persistence**: Save model + scaler + feature ordering metadata

## Privacy

- No raw user images from live sessions are stored
- Only extracted features (27 floats) are used for training
- FER2013 is a public research dataset with explicit CC BY 4.0 license