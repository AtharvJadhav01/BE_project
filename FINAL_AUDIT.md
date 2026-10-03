# Geometric Facial Feature Engineering Experiment - Final Audit

## Audit Summary

**Status:** COMPLETE ✅  
**Date:** 2026-10-02  
**Benchmark:** LinearSVC 50.69% test accuracy (1,449 features: landmarks + blendshapes)

---

## Experiment Results Summary

| Model | Features | Test Accuracy | Macro F1 | Weighted F1 | vs Baseline |
|-------|----------|--------------:| --------: | ----------: | ----------: |
| **LinearSVC** (baseline) | 1,449 | **50.69%** | **49.19%** | **49.98%** | — |
| LogisticRegression | 1,457 | 49.14% | 48.61% | 49.12% | -1.55pp |
| XGBoost | 1,449 | 46.90% | 45.55% | 45.87% | -3.79pp |
| PCA + RBF SVM | 100 PCA | 46.73% | 45.72% | 46.07% | -3.96pp |
| **Geometric (combined)** | 1,503 | 50.60% | 49.29% | 49.81% | -0.09pp |
| geometric_only | 46 | 44.49% | 40.91% | 41.57% | -6.20pp |
| expanded_only | 1,449 | 51.12% | 49.51% | 50.45% | +0.43pp* |

*Note: expanded_only used different validation split; not directly comparable to baseline.*

---

## Key Findings

### ✅ Verified Artifacts
| Artifact | Path | Status |
|----------|------|--------|
| LinearSVC model | `training/models/linear_svc_optimized_1449.joblib` | ✅ Verified |
| LinearSVC scaler | `training/models/scaler_optimized_1449.joblib` | ✅ Verified |
| LinearSVC metadata | `training/models/linear_svc_optimized_1449_metadata.json` | ✅ Verified |
| XGBoost model | `training/models/xgboost_optimized_1449.joblib` | ✅ Verified |
| XGBoost scaler | `training/models/scaler_xgboost_1449.joblib` | ✅ Verified |
| XGBoost metadata | `training/models/xgboost_optimized_1449_metadata.json` | ✅ Verified |
| PCA+RBF SVM pipeline | `training/models/pca_rbf_svm_optimized.joblib` | ✅ Verified |
| PCA+RBF SVM metadata | `training/models/pca_rbf_svm_optimized_metadata.json` | ✅ Verified |
| LogisticRegression model | `training/models/logistic_regression_expanded_1457.joblib` | ✅ Verified |
| Geometric dataset | `training/fer2013_features_geometric/` | ✅ Created |
| Geometric experiment results | `training/geometric_experiment_results.json` | ✅ Saved |
| **Geometric model artifact** | `training/models/geometric_optimized.joblib` | ❌ NOT SAVED |

### Verification Checks
- ✅ Same FER2013 PublicTest split used (1,162 samples)
- ✅ Same 1,449 feature representation for baselines
- ✅ Same 1,503 features for geometric experiments
- ✅ Scaler fitted only on training data
- ✅ Test set never used for model/hyperparameter selection
- ✅ Best configuration selected using validation Macro F1 only
- ✅ LinearSVC artifacts unchanged
- ✅ LogisticRegression artifacts unchanged
- ✅ XGBoost artifacts unchanged
- ✅ PCA+RBF SVM artifacts unchanged
- ✅ No test-set leakage
- ✅ No invented metrics
- ⚠️ Geometric model artifact not saved as separate file
- ⚠️ Geometric feature ablation experiments incomplete

---

## Final Results

| Model | Test Accuracy | Macro F1 | Weighted F1 | vs Baseline |
|-------|--------------:| --------: | ----------: | ----------: |
| **LinearSVC (baseline)** | **50.69%** | **49.19%** | **49.98%** | — |
| LogisticRegression | 49.14% | 48.61% | 49.12% | -1.55pp |
| XGBoost | 46.90% | 45.55% | 45.87% | -3.79pp |
| PCA + RBF SVM | 46.73% | 45.72% | 46.07% | -3.96pp |
| **Geometric (combined)** | 50.60% | 49.29% | 49.81% | **-0.09pp** |
| geometric_only | 44.49% | 40.91% | 41.57% | -6.20pp |
| expanded_only* | 51.12% | 49.51% | 50.45% | +0.43pp* |

*\*expanded_only used different validation split; not directly comparable.*

### Per-Class Performance (Best Geometric Model - Combined)

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|---------|
| angry | 0.340 | 0.290 | 0.313 | 169 |
| disgust | 0.394 | 0.609 | 0.479 | 92 |
| fear | 0.286 | 0.215 | 0.245 | 177 |
| **happy** | **0.760** | **0.720** | **0.739** | 189 |
| sad | 0.331 | 0.263 | 0.293 | 171 |
| surprised | 0.688 | 0.654 | 0.671 | 179 |
| neutral | 0.395 | 0.551 | 0.461 | 185 |

**Difficult classes:** fear (F1=0.245), sad (F1=0.293), angry (F1=0.313)  
**Strong classes:** happy (F1=0.739), surprised (F1=0.671), disgust (F1=0.479)

---

## Final Verdict

**Geometric features do NOT improve upon the verified LinearSVC baseline.**

| Metric | LinearSVC | Geometric (combined) | Difference |
|--------|-----------|---------------------|------------|
| Test Accuracy | 50.69% | 50.60% | **-0.09pp** |
| Macro F1 | 49.19% | 49.29% | **+0.10pp** |
| Weighted F1 | 49.98% | 49.81% | -0.17pp |

**The geometric features provide no meaningful improvement over the LinearSVC baseline.**  
The LinearSVC model (50.69% accuracy) remains the best performing model.

### Recommendation
**Retain the LinearSVC model** as the production model. The geometric features do not provide a meaningful improvement on this dataset and feature representation.

---

## Artifacts Status

| Artifact | Status |
|----------|--------|
| `docs/facial_expression_benchmark_report.md` | ✅ Complete |
| `docs/facial_expression_optimization_report.md` | ✅ Complete |
| `docs/pca_rbf_svm_experiment_report.md` | ✅ Complete |
| `docs/geometric_features_experiment_report.md` | ✅ Complete |
| `docs/xgboost_experiment_report.md` | ✅ Complete |
| All model artifacts | ✅ Verified (except geometric model not saved separately) |

---

## Next Steps (Recommended)

1. **Collect labeled webcam dataset** for real-world validation
2. **Evaluate geometric features** on real-world webcam data
3. **Test temporal modeling** (LSTM/Transformer on landmark sequences)
4. **Cross-dataset evaluation** (RAF-DB, AffectNet, CK+)
5. **Class-balanced training** for fear/sad/angry improvement
6. **Confidence calibration** for reliable probability estimates

---

**Audit Status:** COMPLETE ✅  
**Geometric Features Verified:** No improvement over baseline  
**Recommendation:** Retain LinearSVC as production model