# Medical Sentiment Analysis

A production-ready medical sentiment analysis system using transformer-based models for personalized emotion detection in Polish and English healthcare texts.

## 🚀 Quick Start

### Prerequisites
- Python ≥3.10
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation
```bash
# Clone repository
git clone <repository-url>
cd medical_sentiment_analysis

# Install dependencies
uv sync

# Prepare data and build views
make prepare-data
make build-enc-view
```

### Production Model Usage
```bash
# Use the best performing model (XLM-RoBERTa weighted)
CHECKPOINT=artifacts/models/encoder/enc_baseline_xlmr_weighted make eval-enc-test
```


## 🏗️ Architecture

### Model Pipeline
```
Medical Text → Language Detection → Persona Assignment → 
Tokenization → XLM-RoBERTa → Mean Pooling → 
Linear Head → Binary Classification (18 emotions)
```

### Key Components
- **EncoderClassifier**: Transformer + mean pooling + binary classification head
- **Weighted Loss**: BCEWithLogitsLoss with class-specific weights (0.774-5.451×)
- **Threshold Optimization**: Emotion-specific decision boundaries
- **Persona Integration**: 16 healthcare personas for personalization

## 📁 Project Structure

```
medical_sentiment_analysis/
├── configs/           # YAML configurations
├── modules/           # Core Python modules
├── scripts/           # Training & evaluation scripts
├── artifacts/         # Model checkpoints & results
├── data/             # Processed datasets
├── reports/          # Analysis & evaluation reports
├── archive/          # Deprecated files
└── schema/           # Data validation schemas
```

## 🔧 Development Commands

### Training
```bash
# Train production model
make train-enc-baseline-weighted

# Train with persona tokens
make train-enc-persona-token-xlmr
make train-enc-persona-token-mdeberta
```

### Evaluation
```bash
# Evaluate on test set
make eval-enc-test CHECKPOINT=artifacts/models/encoder/model_name

# Run threshold optimization
python scripts/optimize_thresholds.py

# Generate calibration analysis
make calib-enc CHECKPOINT=artifacts/models/encoder/model_name
```

### Analysis
```bash
# Class imbalance analysis
python scripts/analyze_class_imbalance.py

# Model comparison
python scripts/compare_models.py

# Training visualization
python scripts/visualize_training.py
```

### Healthcare Personas
16 specialized personas including:
- Patient types (young_mother, senior_patient, health_anxiety_patient)
- Providers (supervising_physician, clinic_manager)
- Specialists (mental_health_advocate, healthcare_journalist)

## 📋 Requirements

### System Requirements
- GPU: 8GB+ VRAM for training, 4GB+ for inference
- RAM: 16GB+ recommended
- Storage: 10GB+ for full artifacts

### Dependencies
- transformers ≥4.21.0
- torch ≥1.12.0
- scikit-learn ≥1.1.0
- pandas ≥1.4.0
- numpy ≥1.21.0

## 🔍 Monitoring & Maintenance

### Performance Monitoring
Monitor these metrics in production:
- Per-emotion F1-scores
- Language detection accuracy  
- Persona assignment consistency
- Threshold calibration drift

### Retraining Triggers
- F1-score drops below 60% for Tier 2 emotions
- New medical terminology emergence
- Language distribution changes
- Class imbalance shifts

## 📊 Reports & Documentation

Detailed analysis available in:
- `reports/per_label_analysis_weighted_vs_nonweighted.md`
- `reports/xlmr_vs_mdeberta_comparison_report.md`
- `reports/mdeberta_v3_base_persona_token_evaluation_report.md`
