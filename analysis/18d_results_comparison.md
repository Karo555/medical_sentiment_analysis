# 18D Model Training Results - Baseline vs Fine-tuned Comparison

## Overview
Analysis comparing baseline (pre-trained) vs fine-tuned XLM-RoBERTa performance on the new 18D medical sentiment analysis dataset after removing underrepresented labels.

## Dataset Improvements (21D → 18D)
- **Removed labels**: `embarrassing`, `vulgar`, `more` 
- **Mean imbalance ratio**: 17.2:1 → **4.7:1** (-73%)
- **Max imbalance ratio**: 247:1 → **10.8:1** (-96%)
- **Class weights range**: 0.77-496.0 → **0.77-5.45** (-99% max reduction)

## Test Set Results Comparison

### Overall Performance Metrics

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| **Baseline (Pre-trained)** | 0.4213 | **0.1844** | 0.1309 | **0.5526** |
| **Fine-tuned (Class-weighted)** | **0.7297** | 0.0562 | 0.0455 | 0.0848 |
| **Improvement** | **+73.2%** | -69.5% | -65.3% | -84.7% |

### Key Observations

#### ✅ **Major Accuracy Improvement**:
- **+73.2% accuracy gain** (0.421 → 0.730)
- Fine-tuned model learned to be much more selective
- Significantly reduced false positive rate

#### ⚠️ **Significant Recall Trade-off**:
- **-84.7% recall drop** (0.553 → 0.085)
- Model became overly conservative with positive predictions
- Missing many true positive cases

#### 📊 **Per-Label Analysis**:
**Fine-tuned model only learned to predict 2 out of 18 labels**:
- **negative**: F1=0.678, Precision=0.604, Recall=0.772 ✅
- **happiness**: F1=0.334, Precision=0.215, Recall=0.754 ⚠️
- **All other labels**: F1=0.000 ❌

## Class Weighting Impact

### Improved Weight Distribution (18D):
- **Most weighted labels**:
  - `calm`: 5.45x weight (9.2% positive)
  - `political`: 4.86x weight (10.3% positive)
  - `funny`: 4.43x weight (11.3% positive)
  - `delight`: 4.35x weight (11.5% positive)
  - `inspiring`: 3.73x weight (13.4% positive)

### More Reasonable Weighting:
- **Max weight reduced**: 496x → 5.45x (-99% reduction)
- **Mean weight**: 32.6 → 2.7 (-92% reduction)
- **No extreme weights**: Eliminated impossible-to-learn cases

## Root Cause Analysis

### Why Fine-tuned Model Underperforms:

1. **Still Conservative Due to Weighting**: Even moderate class weights make model cautious
2. **Limited Learning**: Model only converged on 2 most learnable labels
3. **Threshold Issues**: Default 0.5 threshold too high for weighted probabilities
4. **Training Saturation**: Model reached local minimum during training (F1=0 throughout)

## Comparison with Previous 21D Results

| Dataset | Baseline Accuracy | Fine-tuned Accuracy | Baseline F1 | Fine-tuned F1 |
|---------|------------------|-------------------|-------------|---------------|
| **21D** | 0.421 | 0.790 (+87.9%) | 0.166 | 0.147 (-11.4%) |
| **18D** | 0.421 | 0.730 (+73.2%) | 0.184 | 0.056 (-69.5%) |

**Key Insight**: 18D dataset is better balanced but fine-tuned model is still too conservative.

## Recommendations for Improvement

### 1. **Threshold Optimization** 🎯
```python
# Optimize per-label thresholds instead of using 0.5
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.9, 81)
best_thresholds = []
for label_idx in range(18):
    best_f1 = 0
    best_threshold = 0.5
    for threshold in thresholds:
        preds = (probabilities[:, label_idx] > threshold).astype(int)
        f1 = f1_score(y_true[:, label_idx], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    best_thresholds.append(best_threshold)
```

### 2. **Reduce Class Weight Intensity** 📉
```yaml
# Try less aggressive weighting
class_weights_method: "sqrt_inv"  # Instead of "balanced"
class_weights_smooth: 5.0         # Higher smoothing
```

### 3. **Alternative Training Approaches** 🔄
- **Focal Loss**: `FocalLoss(alpha=1, gamma=2)` instead of class weights
- **Macro-averaged F1 Loss**: Direct optimization of F1-score
- **Two-stage Training**: Train without weights first, then fine-tune with light weights

### 4. **Ensemble Approach** 🤝
- Combine baseline (high recall) + fine-tuned (high precision) predictions
- Use weighted voting based on label confidence

## Conclusion

### ✅ **Successes**:
- **Dataset cleanup highly effective**: 73% reduction in imbalance
- **Extreme cases eliminated**: No more impossible-to-learn labels
- **Reasonable class weights**: Max 5.45x vs 496x previously
- **High accuracy achieved**: 73% accuracy gain

### ⚠️ **Challenges Remaining**:
- **Over-conservative learning**: Model too cautious with positive predictions
- **Limited label coverage**: Only 2/18 labels learned effectively
- **Recall-precision tradeoff**: Need better balance

### 🎯 **Next Steps Priority**:
1. **Threshold optimization** (highest impact, lowest effort)
2. **Lighter class weighting** (moderate impact, low effort)  
3. **Focal loss experimentation** (high impact, moderate effort)

The 18D dataset provides a much better foundation for learning, but the training approach needs refinement to achieve the optimal precision-recall balance.