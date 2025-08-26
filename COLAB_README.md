
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
