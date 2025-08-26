# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical sentiment analysis research project that focuses on personalized emotion analysis using transformer-based models. The system supports multiple languages (Polish and English) and incorporates persona-based personalization for healthcare-related text analysis.

The project implements two main approaches:
1. **Encoder-based models** (XLM-RoBERTa, mDeBERTa) with regression heads for 21-dimensional emotion prediction
2. **LLM-based models** with fine-tuning for structured emotion output generation

## Key Architecture

### Data Pipeline
- **Base contract**: `configs/data_base.yaml` defines the canonical data schema with 21 emotion labels
- **Data processing**: Raw medical opinions are preprocessed with persona information and emotion matrices
- **View materialization**: Separate encoder and LLM views are generated from the base data
- **Splits**: Supports language-agnostic (LOLO) and persona-agnostic (LOPO) evaluation strategies

### Model Architecture
- **EncoderRegressor**: `modules/models/encoder_regressor.py` - Transformer backbone + mean pooling + regression head
- **Persona handling**: Three modes - `non_personalized`, `persona_token`, `personalized_desc`
- **Tokenization**: Custom tokenizers with persona-specific tokens in `artifacts/tokenizers/`

### Training Pipeline
- **Trainer**: `modules/training/trainer_encoder.py` - HuggingFace Trainer wrapper with custom metrics
- **Datasets**: `modules/data/datasets.py` - JSONL-based datasets for encoder and LLM views
- **Metrics**: `modules/metrics/` - Regression metrics (R², MAE, etc.) and calibration analysis

## Common Development Commands

### Data Preparation
```bash
# Preview data contracts and smoke tests
make preview-base                    # Test base data contract
make preview-enc                     # Preview encoder view (ENC_MODE=persona_token)
make preview-llm                     # Preview LLM prompts

# Data preprocessing and splits
make prepare-data                    # Process raw data to base format
make make-splits                     # Generate train/val/test splits

# Build materialized views
make build-enc-view                  # Build encoder view (configurable mode)
make build-llm-view                  # Build LLM view
```

### Model Training
```bash
# Baseline encoder training
make train-enc-baseline

# Specific model configurations
make train-enc-persona-token-xlmr        # XLM-R with persona tokens
make train-enc-personalized-xlmr         # XLM-R with persona descriptions
make train-enc-persona-token-mdeberta    # mDeBERTa with persona tokens
make train-enc-personalized-mdeberta     # mDeBERTa with persona descriptions
```

### Evaluation and Analysis
```bash
# Model evaluation (default: validation set)
make eval-enc CHECKPOINT=artifacts/models/encoder/model_name

# Test set evaluation
make eval-enc-test CHECKPOINT=artifacts/models/encoder/model_name

# Calibration analysis
make calib-enc CHECKPOINT=artifacts/models/encoder/model_name
```

### Configuration System
All experiments are defined via YAML configs in `configs/`:
- `experiment/` - Complete experiment configurations
- `model/` - Model architecture definitions
- `data_*.yaml` - Data processing contracts

Key configuration variables:
- `ENC_MODE`: `non_personalized` | `persona_token` | `personalized_desc`
- `LLM_MODE`: `non_personalized` | `persona_token` | `personalized_desc` | `personalized_instruction`
- `CHECKPOINT`: Path to model checkpoint for evaluation
- `EVAL_SPLIT`: `val` | `test`

### Project Structure
- `modules/` - Core Python modules (models, data, training, metrics)
- `scripts/` - Executable scripts for training, evaluation, data processing
- `configs/` - YAML configuration files
- `data/` - Organized data with raw, processed, and split directories
- `artifacts/` - Model checkpoints and custom tokenizers
- `schema/` - JSON schemas for data validation

## Development Notes

### Python Environment
- Uses `uv` for dependency management (see `pyproject.toml`)
- Requires Python ≥3.10
- Key dependencies: transformers, torch, pandas, scikit-learn

### Testing and Validation
- Use `make preview-*` commands for smoke testing before full experiments
- Schema validation is enforced at data processing stages
- All experiments should be reproducible with fixed seeds (default: 1337)

### Multi-language Support
The system is designed for Polish-English bilingual analysis:
- Data contracts specify `lang` field with `pl`/`en` values
- Tokenizers and models handle multilingual input
- Evaluation can be stratified by language

### Persona System
16 healthcare personas are defined in `data/personas/personas.json`:
- Patient personas (young_mother, senior_patient, health_anxiety_patient, etc.)
- Healthcare provider personas (supervising_physician, clinic_manager, etc.)
- Specialized personas (mental_health_advocate, healthcare_journalist, etc.)