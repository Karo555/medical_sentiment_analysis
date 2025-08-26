#!/usr/bin/env python3
"""
Google Colab Setup Helper Script
================================

This script helps prepare the medical sentiment analysis project for training on Google Colab.
Run this locally before uploading to ensure all required files are present.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

def check_required_files() -> Dict[str, List[str]]:
    """Check if all required files for Colab training are present."""
    
    required_files = {
        "data": [
            "data/processed/base/all.jsonl",
            "data/processed/base/sample.jsonl", 
            "data/processed/encoder/train.jsonl",
            "data/processed/encoder/val.jsonl",
            "data/processed/encoder/test.jsonl",
            "data/personas/personas.json"
        ],
        "tokenizers": [
            "artifacts/tokenizers/xlmr-base-with-personas/tokenizer_config.json",
            "artifacts/tokenizers/xlmr-base-with-personas/tokenizer.json",
            "artifacts/tokenizers/mdeberta-v3-base-with-personas/tokenizer_config.json",
            "artifacts/tokenizers/mdeberta-v3-base-with-personas/tokenizer.json"
        ],
        "configs": [
            "configs/experiment/enc_persona_token_xlmr.yaml",
            "configs/experiment/enc_persona_token_mdeberta.yaml", 
            "configs/experiment/enc_personalized_desc_xlmr.yaml",
            "configs/experiment/enc_personalized_desc_mdeberta.yaml",
            "configs/data_base.yaml",
            "configs/data_encoder.yaml"
        ],
        "scripts": [
            "scripts/train_encoder.py",
            "scripts/eval_encoder.py",
            "scripts/preview_base_data.py",
            "scripts/preview_encoder_view.py"
        ],
        "modules": [
            "modules/models/encoder_regressor.py",
            "modules/data/datasets.py",
            "modules/training/trainer_encoder.py",
            "modules/metrics/regression.py",
            "modules/__init__.py",
            "modules/data/__init__.py",
            "modules/models/__init__.py",
            "modules/training/__init__.py",
            "modules/metrics/__init__.py"
        ]
    }
    
    status = {}
    for category, files in required_files.items():
        missing = []
        present = []
        
        for file_path in files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                present.append(f"{file_path} ({size:,} bytes)")
            else:
                missing.append(file_path)
        
        status[category] = {
            "present": present,
            "missing": missing
        }
    
    return status

def generate_requirements_summary() -> Dict[str, Any]:
    """Generate a summary of project requirements."""
    
    # Check dependencies
    try:
        import yaml
        with open("pyproject.toml", "r") as f:
            pyproject_content = f.read()
    except Exception as e:
        pyproject_content = f"Error reading pyproject.toml: {e}"
    
    # Check data statistics
    data_stats = {}
    try:
        # Count samples in each split
        for split in ["train", "val", "test"]:
            file_path = f"data/processed/encoder/{split}.jsonl"
            if Path(file_path).exists():
                with open(file_path, "r") as f:
                    count = sum(1 for _ in f)
                data_stats[split] = count
    except Exception as e:
        data_stats = {"error": str(e)}
    
    return {
        "dependencies": pyproject_content,
        "data_statistics": data_stats,
        "gpu_requirements": "CUDA-compatible GPU (T4, V100, A100 recommended)",
        "memory_requirements": "~4-8 GB GPU memory for training",
        "estimated_training_time": {
            "xlmr_persona_token": "30-60 minutes (3 epochs)",
            "mdeberta_persona_token": "45-90 minutes (3 epochs)",
            "full_experiment_suite": "3-6 hours (all 4 experiments)"
        }
    }

def create_colab_readme() -> str:
    """Create a README for Colab setup."""
    
    readme_content = """
# Medical Sentiment Analysis - Google Colab Training

## Quick Start

1. **Upload to Google Colab**:
   - Open Google Colab (colab.research.google.com)
   - Upload `Medical_Sentiment_Analysis_Training.ipynb`
   - Make sure to select GPU runtime: Runtime → Change runtime type → Hardware accelerator: GPU

2. **Update Repository URL**:
   - In the notebook, update `REPO_URL` variable with your actual repository URL
   - Ensure your repository is public or you have proper authentication set up

3. **Run the Notebook**:
   - Execute cells sequentially
   - First-time setup will take 5-10 minutes
   - Training will take 30-90 minutes per experiment

## Experiments Available

- `xlmr_persona_token`: XLM-RoBERTa with persona tokens
- `mdeberta_persona_token`: mDeBERTa with persona tokens  
- `xlmr_personalized`: XLM-RoBERTa with persona descriptions
- `mdeberta_personalized`: mDeBERTa with persona descriptions

## Expected Results

Each experiment will produce:
- Trained model checkpoints
- Evaluation metrics (R², MAE, RMSE, Spearman)
- Training plots and logs
- Results summary JSON

## Troubleshooting

- **GPU not available**: Enable GPU runtime in Colab settings
- **Out of memory**: Reduce batch sizes in the notebook
- **Missing files**: Ensure all data and tokenizer files are committed to repository
- **Module import errors**: Check PYTHONPATH is set correctly

## File Storage

All outputs are automatically saved to your Google Drive in:
`MyDrive/medical_sentiment_analysis/training_runs/TIMESTAMP/`
"""
    
    return readme_content

def main():
    """Main setup check function."""
    
    print("🔍 Checking Google Colab Setup Requirements...")
    print("=" * 60)
    
    # Check required files
    status = check_required_files()
    
    total_missing = 0
    for category, files in status.items():
        print(f"\n📁 {category.upper()}:")
        
        if files["present"]:
            print(f"  ✅ Present ({len(files['present'])} files):")
            for file in files["present"][:3]:  # Show first 3
                print(f"    - {file}")
            if len(files["present"]) > 3:
                print(f"    ... and {len(files['present']) - 3} more")
        
        if files["missing"]:
            print(f"  ❌ Missing ({len(files['missing'])} files):")
            for file in files["missing"]:
                print(f"    - {file}")
            total_missing += len(files["missing"])
    
    print(f"\n📊 Summary:")
    print(f"  Missing files: {total_missing}")
    
    if total_missing == 0:
        print("  🎉 All required files are present!")
        print("  ✅ Ready for Google Colab training!")
    else:
        print("  ⚠️  Some files are missing. Please run data preparation steps:")
        print("     - make prepare-data")
        print("     - make build-enc-view") 
        print("     - Ensure custom tokenizers are built")
    
    # Generate requirements summary
    requirements = generate_requirements_summary()
    
    print(f"\n🔧 Requirements Summary:")
    print(f"  Data samples: {requirements['data_statistics']}")
    print(f"  GPU: {requirements['gpu_requirements']}")
    print(f"  Memory: {requirements['memory_requirements']}")
    
    print(f"\n⏱️  Estimated Training Times:")
    for exp, time in requirements['estimated_training_time'].items():
        print(f"  {exp}: {time}")
    
    # Create helper files
    colab_readme = create_colab_readme()
    with open("COLAB_README.md", "w") as f:
        f.write(colab_readme)
    
    with open("colab_requirements.json", "w") as f:
        json.dump({
            "file_status": status,
            "requirements": requirements
        }, f, indent=2)
    
    print(f"\n📝 Created helper files:")
    print(f"  - COLAB_README.md")
    print(f"  - colab_requirements.json")
    print(f"  - Medical_Sentiment_Analysis_Training.ipynb")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Fix any missing files")
    print(f"  2. Update REPO_URL in the notebook")
    print(f"  3. Upload notebook to Google Colab")
    print(f"  4. Enable GPU runtime and start training!")

if __name__ == "__main__":
    main()