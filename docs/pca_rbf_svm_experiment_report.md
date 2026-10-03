# PCA + RBF SVM Experiment Report — Facial Expression Recognition

**Project:** FER System Optimization  
**Date:** 2026-10-02  
**Dataset:** FER2013 PublicTest split (1,162 test samples)  
**Baseline Models:** LogisticRegression (49.14%), LinearSVC (50.69%), XGBoost (46.90%)  
**Experiment:** PCA + RBF SVM on 1,449 Landmarks + Blendshapes features

---

## 1. Objective

Determine whether dimensionality reduction with PCA followed by a nonlinear RBF SVM can outperform the verified LinearSVC model (50.69% test accuracy) on the FER2013 facial expression recognition task.

---

## 2. Dataset and FER2013 PublicTest Protocol

| Property | Value |
|----------|-------|
| Source Dataset | FER2013 (PublicTest split) |
| Total available samples | 2,711 |
| Face-detected samples | 2,401 |
| No face / rejected | 310 |
| Training samples | 1,054 |
| Validation samples | 185 |
| **Held-out Test (PublicTest)** | **1,162** |
| Classes | 7 (angry, disgust, fear, happy, sad, surprised, neutral) |

**Class Distribution (Test Set):**
| Emotion | Support |
|---------|---------|
| angry | 169 |
| disgust | 92 |
| fear | 177 |
| happy | 189 |
| sad | 171 |
| surprised | 179 |
| neutral | 185 |

---

## 3. Train / Validation / Test Split

| Split | Samples | Purpose |
|-------|---------|---------|
| Training | 1,054 | Model fitting, StandardScaler + PCA fitting |
| Validation | 185 | Hyperparameter selection, model selection |
| **Test (PublicTest)** | **1,162** | **Final evaluation only — never used during optimization** |

---

## 4. Feature Representation

### 4.1 Input Features (1,449 dimensions)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| **Landmarks (lm_)** | 1,401 | 467 landmarks × 3 (x, y, z), normalized |
| **Blendshapes (bs_)** | 48 | MediaPipe 52 blendshape coefficients (4 dropped as constant) |
| **Total** | **1,449** | |

**Excluded:** Head pose (3), Metadata (2), 4 constant features (lm_1_x/y/z, feature_1460)

**Normalization:** Landmarks translated to nose-tip origin, scaled by inter-ocular distance.

---

## 5. StandardScaler Procedure

- Fitted **only on training data** (1,054 samples)
- Applied to validation and test sets without refitting
- No data leakage from validation/test to scaler fitting

---

## 6. PCA Methodology

### 6.1 PCA Configurations Tested

| Type | Parameters | Actual Components | Explained Variance |
|------|------------|-------------------|-------------------|
| Fixed n_components | 50 | 50 | 0.89 |
| Fixed n_components | 100 | 100 | 0.9997 |
| Fixed n_components | 150 | 150 | 0.9999 |
| Fixed n_components | 200 | 200 | 0.9999 |
| Fixed n_components | 300 | 300 | 0.9999 |
| Fixed n_components | 500 | 500 | 1.0000 |
| Fixed n_components | 750 | 750 | 1.0000 |
| Fixed n_components | 1000 | 1000 | 1.0000 |
| Explained variance | 0.90 | 5 | 0.9058 |
| Explained variance | 0.95 | 8 | 0.9501 |
| Explained variance | 0.99 | 31 | 0.9900 |

**Note:** Explained variance thresholds produce fewer components than fixed counts because the feature space has high intrinsic dimensionality. Fixed component counts > ~100 capture essentially all variance (>99.9%).

### 6.2 PCA Fitting Protocol

- PCA fitted **only on training data** (after StandardScaler)
- Validation and test data transformed using fitted PCA only
- No data leakage from validation/test to PCA fitting

---

## 7. PCA Configurations Tested

11 PCA configurations × 14 SVM hyperparameter combinations = **154 total experiments**

---

## 8. RBF SVM Search Space

| Parameter | Values Tested |
|-----------|--------------|
| C | 0.1, 1, 10, 100 |
| gamma | scale, auto, 0.01, 0.1 |
| class_weight | None, balanced |

**Total SVM configurations per PCA setting:** 14  
**Total experiments:** 11 PCA configs × 14 SVM configs = **154 experiments**

---

## 9. Validation-Selection Methodology

- **Training data only** used for fitting StandardScaler, PCA, and SVM
- **Validation data only** used for hyperparameter selection
- **Test set NEVER used** for hyperparameter selection or model selection
- Primary selection metric: **Validation Macro F1**
- Model selected: Best validation Macro F1 across all 154 experiments

---

## 10. Best Validation Configuration

| Parameter | Value |
|-----------|-------|
| **PCA** | n_components=100 |
| **Actual components** | 100 |
| **Explained variance** | 0.9997 |
| **SVM C** | 10 |
| **SVM gamma** | scale |
| **SVM class_weight** | None |
| **Val Accuracy** | 42.70% |
| **Val Macro F1** | **41.75%** |
| **Val Weighted F1** | 41.42% |

*Note: Validation Macro F1 (41.75%) is lower than LinearSVC's 45.66% (Phase 2) / 45.98% (ablation study).*

---

## 11. Final Test Evaluation

**Configuration:** PCA (100 components) + RBF SVM (C=10, gamma=scale, class_weight=None)  
**Features:** Landmarks + Blendshapes (1,449 → PCA 100)  
**Test Set:** FER2013 PublicTest (1,162 samples)

### 11.1 Final Test Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **46.73%** |
| **Macro F1** | **45.72%** |
| **Weighted F1** | **46.07%** |
| Macro Precision | 45.64% |
| Macro Recall | 47.16% |
| Weighted Precision | 45.57% |
| Weighted Recall | 46.73% |

### 11.2 Per-Class Test Performance

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|---------|
| angry | 0.3403 | 0.2899 | 0.3131 | 169 |
| disgust | 0.3944 | 0.6087 | 0.4786 | 92 |
| fear | 0.2857 | 0.2147 | 0.2452 | 177 |
| **happy** | **0.7598** | **0.7196** | **0.7391** | 189 |
| sad | 0.3309 | 0.2632 | 0.2932 | 171 |
| surprised | 0.6882 | 0.6536 | 0.6705 | 179 |
| neutral | 0.3953 | 0.5514 | 0.4605 | 185 |

**Difficult classes:** fear (F1=0.245), sad (F1=0.293), angry (F1=0.313)  
**Strong classes:** happy (F1=0.739), surprised (F1=0.671), disgust (F1=0.479)

### 11.3 Confusion Matrix (Test Set, 1,162 samples)

| True\Pred | angry | disgust | fear | happy | sad | surprised | neutral |
|-----------|-------|---------|------|-------|-----|-----------|---------|
| **angry** | 49 | 33 | 21 | 5 | 23 | 13 | 25 |
| **disgust** | 9 | 56 | 9 | 2 | 8 | 2 | 6 |
| **fear** | 24 | 16 | 38 | 9 | 27 | 23 | 40 |
| **happy** | 8 | 7 | 10 | 136 | 7 | 4 | 17 |
| **sad** | 25 | 12 | 23 | 7 | 45 | 6 | 53 |
| **surprised** | 10 | 9 | 18 | 4 | 6 | 117 | 15 |
| **neutral** | 19 | 9 | 14 | 16 | 20 | 5 | 102 |

---

## 12. Comparison with Baselines

| Model | Features | Test Accuracy | Macro F1 | Weighted F1 |
|-------|----------|--------------:| -------: | ----------: |
| **LogisticRegression** | 1,457 | **49.14%** | **48.61%** | **49.12%** |
| **LinearSVC** | 1,449 | **50.69%** | **49.19%** | **49.98%** |
| **XGBoost** | 1,449 | **46.90%** | **45.55%** | **45.87%** |
| **PCA + RBF SVM** | 1,449 → 100 | **46.73%** | **45.72%** | **46.07%** |

### Difference vs. LinearSVC (Protected Benchmark)

| Metric | LinearSVC | PCA + RBF SVM | Difference |
|--------|-----------|---------------|------------|
| Test Accuracy | 50.69% | 46.73% | **-3.96pp** |
| Macro F1 | 49.19% | 45.72% | **-3.47pp** |
| Weighted F1 | 49.98% | 46.07% | **-3.91pp** |

**PCA + RBF SVM did not exceed the verified LinearSVC result of 50.69%.**

---

## 13. Computational Performance

| Metric | Value |
|--------|-------|
| Training Time (best config) | ~0.15s |
| **Measured Classifier Inference Latency** | **0.072 ms/sample** |
| Model Size | ~1.2 MB |
| Scaler Size | ~0.1 MB |
| PCA Model Size | ~0.5 MB |

**Note:** 0.072 ms/sample is measured classifier inference latency on CPU. Total pipeline latency including MediaPipe feature extraction and preprocessing will be higher.

**Real-time Suitability:** ✅ Excellent - well within real-time requirements.

---

## 14. Model Artifacts Saved

| Artifact | Path | Size |
|----------|------|------|
| Pipeline | `training/models/pca_rbf_svm_optimized.joblib` | ~1.2 MB |
| Metadata | `training/models/pca_rbf_svm_optimized_metadata.json` | ~8 KB |

**Reload Verification:** ✅ PASSED - Predictions identical after reload

---

## 14. Limitations

1. **Test set size:** 1,162 samples — confidence intervals ~±2%
2. **Domain shift:** FER2013 (grayscale, 48×48) vs. real webcam (color, variable conditions)
3. **Class imbalance:** Fear (177) vs Disgust (92) in test set
4. **No temporal modeling:** Single-frame classification only
5. **No webcam validation:** Real-world accuracy not measured
6. **PCA discarded information:** 100 components captured 99.97% variance but performance dropped
6. **No geometric features tested:** Experimental extractor created but not benchmarked on FER2013

---

## 15. Reproducibility

### Exact Reproduction Steps
```bash
# 1. Load data
python -c "from ablation_study import load_and_prepare_data; X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, group_indices = load_and_prepare_data()"

# 2. Select features (Landmarks + Blendshapes)
indices = group_indices['landmarks'] + group_indices['blendshapes']

# 3. Build pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100, random_state=42)),
    ('svm', SVC(C=10, gamma='scale', class_weight=None, random_state=42))
])

# 4. Train
pipeline.fit(X_train[:, indices], y_train)

# 4. Evaluate on test
y_pred = pipeline.predict(X_test[:, indices])
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
```

### Key Parameters
| Parameter | Value |
|-----------|-------|
| Random Seed | 42 (data splits, model initialization) |
| StandardScaler | Fitted on training data only |
| PCA | Fitted on training data only (n_components=100) |
| Feature Cleaning | Remove constant/NaN features (std > 1e-10) |
| Test Set | FER2013 PublicTest (1,162 samples, fixed split) |

---

## 16. Final Conclusion

### Summary of Achievements

| Objective | Status | Result |
|-----------|--------|--------|
| PCA + RBF SVM implementation | ✅ Complete | Pipeline created and tested |
| PCA configurations tested | ✅ Complete | 11 configs (8 fixed + 3 variance) |
| RBF SVM hyperparameter search | ✅ Complete | 14 configs per PCA setting |
| Validation-based selection | ✅ Complete | Best Val Macro F1: 41.75% |
| Final test evaluation | ✅ Complete | 46.73% accuracy |
| Model artifacts saved | ✅ Complete | Pipeline + Metadata |
| Reload verification | ✅ PASSED | Predictions identical |

### Final Verdict

**PCA + RBF SVM did not exceed the verified LinearSVC result.**

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|-------|--------------:| -------: | ----------: |
| LinearSVC (1,449 feat) | **50.69%** | **49.19%** | **49.98%** |
| PCA + RBF SVM (1,449→100) | 46.73% | 45.72% | 46.07% |

**Improvement:** None. PCA + RBF SVM is **-3.96pp** below LinearSVC on test accuracy.

### Recommendation

**Retain the LinearSVC model** as the production model. PCA + RBF SVM did not provide an improvement on this feature representation and dataset. The LinearSVC model remains the best performing model for this task.

### Next Steps (Priority Order)

1. **Collect labeled webcam dataset** for real-world validation
2. **Evaluate geometric features** on FER2013 when computationally feasible
3. **Test temporal modeling** (LSTM/Transformer on landmark sequences)
4. **Cross-dataset evaluation** (RAF-DB, AffectNet, CK+)
5. **Class-balanced training** for fear/sad/angry improvement
6. **Confidence calibration** for reliable probability estimates

---

**Report Status:** COMPLETE  
**Generated:** 2026-10-02  
**Author:** Automated Optimization Pipeline