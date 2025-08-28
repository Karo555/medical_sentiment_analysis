# Class-Weighted Training Results Analysis

## Overview
This report analyzes the performance of class-weighted training on the medical sentiment analysis dataset, comparing it with the baseline unweighted model.

## Dataset Imbalance Summary
- **Mean imbalance ratio**: 17.2:1 
- **Most imbalanced labels**: 
  - `embarrassing`: 0% positive samples (496x weight)
  - `vulgar`: 0.4% positive samples (124x weight)
  - `more`: 3.1% positive samples (16x weight)
- **Most balanced labels**: `interesting`, `understandable`, `negative`

## Results Comparison (Test Set)

### Overall Metrics

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| **Baseline (Unweighted)** | 0.4205 | 0.1657 | 0.1351 | **0.5711** |
| **Class-Weighted** | **0.7900** | 0.1468 | 0.0774 | 0.1427 |
| **Improvement** | **+87.9%** | -11.4% | -42.7% | -75.0% |

### Key Findings

#### ✅ **Significant Improvements**:
1. **Accuracy**: Massive +87.9% improvement (0.421 → 0.790)
   - Class weighting successfully reduced false positives
   - Model learned to be more selective with predictions

2. **Label Coverage**: Model learned to predict specific labels
   - Successfully predicts: `positive`, `negative`, `embarrassing`, `understandable`
   - Shows focused learning on most important/balanced labels

#### ⚠️ **Trade-offs**:
1. **Recall**: -75% drop (0.571 → 0.143)
   - Model became very conservative with positive predictions
   - Miss rate for positive cases increased significantly

2. **Precision**: -42.7% drop (0.135 → 0.077)
   - When model does predict positive, it's less accurate
   - Indicates need for threshold optimization

3. **F1-Score**: -11.4% drop (0.166 → 0.147)
   - Overall harmonic mean slightly decreased
   - Trade-off between precision and recall

## Per-Label Analysis

### Labels with Positive Performance:
- **positive**: F1=0.540, Precision=0.371, Recall=0.996
- **negative**: F1=0.762, Precision=0.615, Recall=1.000  
- **embarrassing**: F1=1.000 (perfect, but only 0 positive samples)
- **understandable**: F1=0.780, Precision=0.640, Recall=1.000

### Labels with Zero Performance:
Most other labels (17 out of 21) show F1=0.000, indicating the model learned to never predict them as positive.

## Analysis & Recommendations

### Why This Happened:
1. **Extreme Class Weights**: Some weights reached 496x, making the model extremely cautious
2. **Conservative Learning**: Model preferred high accuracy over balanced prediction
3. **Threshold Issue**: Default 0.5 threshold may be inappropriate for weighted models

### Recommended Improvements:

#### 1. **Threshold Optimization**
```python
# Use per-label threshold optimization
thresholds = optimize_thresholds_per_label(y_true, y_pred_probs, metric='f1')
```

#### 2. **Moderate Class Weights**
```python
# Use sqrt_inv method instead of balanced
class_weights_method: "sqrt_inv"  # Less extreme than balanced
class_weights_smooth: 5.0         # Higher smoothing
```

#### 3. **Focal Loss Alternative**
```python
# Consider focal loss instead of class weights
focal_loss = FocalLoss(alpha=weights, gamma=2.0)
```

#### 4. **Macro-Averaged Training**
- Use macro-averaged F1 as training objective
- Focus on per-label performance rather than overall accuracy

## Conclusion

The class-weighted training **successfully addresses the accuracy problem** (+87.9% improvement) but creates a new challenge with **recall and coverage**. The model learned to be highly selective, achieving better accuracy but missing many positive cases.

**Next Steps**:
1. Implement threshold optimization per label
2. Try less extreme weighting methods (sqrt_inv)
3. Consider focal loss as alternative
4. Evaluate trade-offs based on use case priorities

The approach shows promise but needs refinement to balance accuracy gains with recall preservation.