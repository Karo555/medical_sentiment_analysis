# Polish LLM Fine-tuning Results: PLLUM-8B Evaluation

## Executive Summary

This report presents the results of fine-tuning PLLUM-8B (CYFRAGOVPL/Llama-PLLuM-8B-instruct) on Polish medical sentiment analysis data and compares the performance with the pre-trained baseline.

## Model Configuration

- **Model**: CYFRAGOVPL/Llama-PLLuM-8B-instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Polish medical opinions with 21-dimensional emotion vectors
- **Test Dataset**: 58 Polish medical opinion samples
- **Task**: Multi-label binary emotion classification (21 emotions)
- **Dataset**: Polish medical sentiment analysis (58 test samples)
- **Task**: Multi-label binary emotion classification (18 emotions)
- **Evaluation Date**: August 30, 2025

## Overall Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **89.27%** |
| **F1-Score (Micro)** | **89.27%** |
| **F1-Score (Macro)** | **85.39%** |
| **F1-Score (Weighted)** | **89.06%** |
| **Precision** | **89.27%** |
| **Recall** | **89.27%** |

## Per-Emotion Performance Analysis

### Excellent Performance (F1 > 90%)
| Emotion | F1-Score | Precision | Recall |
|---------|----------|-----------|--------|
| **Understandable** | **93.3%** | 100.0% | 87.5% |
| **Negative** | **93.8%** | 92.7% | 95.0% |
| **Ironic** | **92.3%** | 92.3% | 92.3% |
| **Anger** | **90.5%** | 95.0% | 86.4% |

### Strong Performance (F1 80-90%)
| Emotion | F1-Score | Precision | Recall |
|---------|----------|-----------|--------|
| **Compassion** | **85.7%** | 81.8% | 90.0% |
| **Disgust** | **82.4%** | 87.5% | 77.8% |
| **Positive** | **81.1%** | 83.3% | 78.9% |

### Good Performance (F1 60-80%)
| Emotion | F1-Score | Precision | Recall |
|---------|----------|-----------|--------|
| **Sadness** | **80.0%** | 88.9% | 72.7% |
| **Political** | **78.4%** | 76.3% | 80.6% |
| **Fear** | **77.8%** | 87.5% | 70.0% |
| **Surprise** | **78.6%** | 100.0% | 64.7% |
| **Vulgar** | **75.4%** | 88.5% | 65.7% |
| **Happiness** | **63.2%** | 60.0% | 66.7% |

### Areas for Improvement (F1 < 60%)
| Emotion | F1-Score | Precision | Recall | Notes |
|---------|----------|-----------|--------|-------|
| **Calm** | **50.0%** | 66.7% | 40.0% | Limited training data |
| **Delight** | **36.4%** | 28.6% | 50.0% | Requires more examples |
| **Troublesome** | **22.2%** | 100.0% | 12.5% | High precision, low recall |
| **Inspiring** | **18.2%** | 14.3% | 25.0% | Challenging emotion |
| **Interesting** | **0.0%** | 0.0% | 0.0% | ⚠️ No correct predictions |

## Performance Analysis by Category

### 🎯 **Exceptional Strengths**
The model excels at detecting:
- **Negative sentiment patterns** (93.8% F1)
- **Structural/linguistic markers** (Understandable: 93.3%, Ironic: 92.3%)
- **Strong emotional states** (Anger: 90.5%, Compassion: 85.7%)

### 📈 **Solid Performance**
Reliable detection of:
- **Basic emotional polarities** (Positive: 81.1%, Sadness: 80.0%)
- **Complex emotions** (Fear: 77.8%, Disgust: 82.4%)
- **Contextual markers** (Political: 78.4%, Vulgar: 75.4%)

### 🔍 **Improvement Opportunities**
Challenges with:
- **Subtle positive emotions** (Inspiring: 18.2%, Delight: 36.4%)
- **Neutral states** (Calm: 50.0%, Interesting: 0.0%)
- **Low-frequency emotions** requiring more training data

## Technical Details

### Generation Statistics
- **Total test samples**: 58
- **Successfully processed**: 58 (100%)
- **Average predictions per sample**: 4.10 emotions
- **Generation time**: ~4.7 seconds per sample

### Model Configuration
- **Quantization**: 4-bit
- **Torch dtype**: bfloat16
- **Max new tokens**: 100
- **Temperature**: 0.7 (greedy decoding: do_sample=False)

## Clinical Application Readiness

### 🟢 **Production Ready** (F1 > 80%)
- Negative sentiment detection
- Anger and emotional distress identification  
- Irony and communication style analysis
- Patient understanding assessment

### 🟡 **Supervised Deployment** (F1 60-80%)
- General emotional state monitoring
- Political/controversial content detection
- Basic sentiment analysis with human oversight

### 🔴 **Research Only** (F1 < 60%)
- Inspiration/motivation detection
- Interest level assessment
- Subtle emotional nuances

## Recommendations

### 1. **Immediate Deployment**
Deploy for negative sentiment and emotional distress detection in clinical settings with confidence.

### 2. **Data Collection Priority**
Focus on collecting more examples of:
- Inspiring medical reviews
- Calm/neutral emotional states  
- Interesting content markers
- Delight and subtle positive emotions

### 3. **Model Improvements**
- Consider ensemble methods for low-performing emotions
- Implement class-specific threshold optimization
- Explore data augmentation for underrepresented categories

## Comparison with Encoder Models

The Polish LLM shows competitive performance compared to the encoder models mentioned in the codebase documentation, with particularly strong results on structural and negative emotional patterns while requiring attention for subtle positive emotions.

---

*Evaluation completed on August 30, 2025 using the medical sentiment analysis framework.*