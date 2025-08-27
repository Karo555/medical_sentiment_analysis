# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical sentiment analysis research project that focuses on personalized emotion analysis using transformer-based models. The system supports multiple languages (Polish and English) and incorporates persona-based personalization for healthcare-related text analysis.

The project implements two main approaches:
1. **Encoder-based models** (XLM-RoBERTa, mDeBERTa) with regression heads for 21-dimensional emotion prediction
2. **LLM-based models** with LoRA fine-tuning for structured emotion output generation (Gemma, Mistral, Qwen)

## Key Architecture

### Data Pipeline
- **Base contract**: `configs/data_base.yaml` defines the canonical data schema with 21 emotion labels
- **Data processing**: Raw medical opinions are preprocessed with persona information and emotion matrices
- **View materialization**: Separate encoder and LLM views are generated from the base data
- **Splits**: Supports language-agnostic (LOLO) and persona-agnostic (LOPO) evaluation strategies

### Model Architecture
- **EncoderRegressor**: `modules/models/encoder_regressor.py` - Transformer backbone + mean pooling + regression head
- **LLMTrainer**: `modules/training/trainer_llm.py` - Custom trainer with LoRA support and generation-based evaluation
- **Persona handling**: Four modes - `non_personalized`, `persona_token`, `personalized_desc`, `personalized_instruction`
- **Tokenization**: Custom tokenizers with persona-specific tokens in `artifacts/tokenizers/`

### Training Pipeline
- **Encoder Trainer**: `modules/training/trainer_encoder.py` - HuggingFace Trainer wrapper with custom regression metrics
- **LLM Trainer**: `modules/training/trainer_llm.py` - Enhanced trainer with LoRA, quantization, and generation evaluation
- **Datasets**: `modules/data/datasets.py` - JSONL-based datasets for encoder and LLM views with flexible tokenization
- **Collate Functions**: `modules/data/collate.py` - Specialized batch processing for encoder and LLM training
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

#### Encoder Training
```bash
# Baseline encoder training
make train-enc-baseline

# Specific model configurations
make train-enc-persona-token-xlmr        # XLM-R with persona tokens
make train-enc-personalized-xlmr         # XLM-R with persona descriptions
make train-enc-persona-token-mdeberta    # mDeBERTa with persona tokens
make train-enc-personalized-mdeberta     # mDeBERTa with persona descriptions
```

#### LLM Training
```bash
# Individual LLM model training
make train-llm-baseline-gemma2           # Gemma2-2B baseline (non-personalized)
make train-llm-persona-token-mistral     # Mistral-7B with persona tokens + LoRA
make train-llm-personalized-desc-qwen2   # Qwen2-1.5B with persona descriptions + LoRA  
make train-llm-gemma2-27b-personalized   # Gemma2-27B-IT with persona descriptions + LoRA

# Complete LLM workflows (evaluation + training + visualization + comparison)
make workflow-gemma2-27b-complete        # Full workflow with comprehensive evaluation
make workflow-gemma2-27b-quick           # Fast workflow with reduced evaluation samples
```

### Evaluation and Analysis

#### Encoder Evaluation
```bash
# Model evaluation (default: validation set)
make eval-enc CHECKPOINT=artifacts/models/encoder/model_name

# Test set evaluation
make eval-enc-test CHECKPOINT=artifacts/models/encoder/model_name

# Calibration analysis
make calib-enc CHECKPOINT=artifacts/models/encoder/model_name
```

#### LLM Evaluation
```bash
# Baseline model evaluation (pre-training)
make eval-llm-gemma2-baseline             # Evaluate pre-trained Gemma2-27B

# Fine-tuned model evaluation (post-training) 
make eval-llm-gemma2-trained              # Evaluate fine-tuned Gemma2-27B

# Training visualization and analysis
make visualize-llm-training CHECKPOINT=artifacts/models/llm/model_name
```

### Configuration System
All experiments are defined via YAML configs in `configs/`:
- `experiment/` - Complete experiment configurations (encoder and LLM)
- `model/` - Model architecture definitions (including LoRA configurations)
- `data_*.yaml` - Data processing contracts for encoder and LLM views

#### Key Configuration Variables:
- `ENC_MODE`: `non_personalized` | `persona_token` | `personalized_desc`
- `LLM_MODE`: `non_personalized` | `persona_token` | `personalized_desc` | `personalized_instruction`
- `CHECKPOINT`: Path to model checkpoint for evaluation
- `EVAL_SPLIT`: `val` | `test`

#### LLM-Specific Configurations:
- **Model Settings**: Quantization (4-bit/8-bit), flash attention, device mapping
- **LoRA Settings**: Rank (r), alpha, dropout, target modules
- **Training Settings**: Learning rates, batch sizes, gradient accumulation
- **Generation Settings**: Max tokens, temperature, sampling parameters

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
- Key dependencies: transformers, torch, pandas, scikit-learn, peft, bitsandbytes, accelerate

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

### LLM Training Features
The project includes comprehensive LLM fine-tuning capabilities:

#### Supported Models
- **Gemma 2**: 2B, 27B-IT (instruction-tuned) variants
- **Mistral**: 7B-v0.3 with LoRA optimization
- **Qwen2**: 1.5B multilingual model
- **Custom models**: Extensible configuration system

#### Advanced Training Features
- **LoRA Fine-tuning**: Memory-efficient training with configurable ranks and targets
- **Quantization**: 4-bit and 8-bit quantization for large model training
- **Generation Evaluation**: Real-time JSON parsing and emotion prediction metrics
- **Enhanced Monitoring**: Detailed progress tracking, time estimates, and parameter statistics
- **Checkpoint Management**: Automatic saving and resuming with best model selection

#### Authentication Setup
For gated models (like Gemma), set up authentication:
1. Create `.env` file in project root
2. Add your token: `HF_TOKEN=hf_your_token_here` 
3. Accept model licenses on HuggingFace Hub
4. The system automatically handles authentication during training

#### Training Workflows
- **Individual Training**: Train specific models with custom configurations
- **Complete Workflows**: Automated baseline evaluation → training → post-evaluation → visualization
- **Comparison Analysis**: Automatic performance comparison between pre/post training models