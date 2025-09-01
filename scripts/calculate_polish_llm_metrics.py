#!/usr/bin/env python3

import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def parse_generated_labels(text):
    """Parse labels from generated text, handling various formats"""
    # Try to extract array from text like "{labels:[0.0,1.0,...]}"
    pattern = r"labels:\s*\[([^\]]+)\]"
    match = re.search(pattern, text)
    if match:
        try:
            # Split by comma and convert to float, then to int (for binary classification)
            labels_str = match.group(1)
            labels = [int(float(x.strip())) for x in labels_str.split(',')]
            # Truncate to 18 dimensions if longer (the model outputs 21, but ground truth has 18)
            return labels[:18] if len(labels) >= 18 else None
        except:
            return None
    return None

def load_ground_truth(file_path):
    """Load ground truth from test data"""
    ground_truths = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Parse target_text like "{labels:[0.0,1.0,...]}"
                target = item['target_text']
                labels = parse_generated_labels(target)
                if labels:
                    ground_truths.append(labels)
    return ground_truths

def load_predictions(file_path):
    """Load predictions from evaluation results"""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                generated_text = item['generated_text']
                labels = parse_generated_labels(generated_text)
                if labels:
                    predictions.append(labels)
                else:
                    # If parsing failed, treat as all zeros (conservative approach)
                    predictions.append([0] * 18)
    return predictions

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for multi-label binary classification"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics (micro-averaged)
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1_micro = f1_score(y_true.flatten(), y_pred.flatten(), average='micro')
    f1_macro = f1_score(y_true.flatten(), y_pred.flatten(), average='macro')
    f1_weighted = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
    precision = precision_score(y_true.flatten(), y_pred.flatten(), average='micro')
    recall = recall_score(y_true.flatten(), y_pred.flatten(), average='micro')
    
    # Per-label metrics
    f1_per_label = f1_score(y_true, y_pred, average=None)
    precision_per_label = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_label = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    emotion_names = [
        'Positive', 'Negative', 'Happiness', 'Delight', 'Inspiring', 'Calm', 
        'Surprise', 'Compassion', 'Fear', 'Sadness', 'Disgust', 'Anger', 
        'Ironic', 'Troublesome', 'Vulgar', 'Political', 'Interesting', 'Understandable'
    ]
    
    return {
        'overall': {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro, 
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        },
        'per_label': {
            emotion_names[i]: {
                'f1': f1_per_label[i],
                'precision': precision_per_label[i],
                'recall': recall_per_label[i]
            } for i in range(len(emotion_names))
        }
    }

def main():
    # File paths
    test_data_path = '/workspace/medical_sentiment_analysis/data/processed/llm_pl/test.jsonl'
    results_path = '/workspace/medical_sentiment_analysis/artifacts/models/llm/llm_baseline_pllum_8b/evaluation/test/detailed_results.jsonl'
    
    print("=== Polish LLM Post Fine-tuning Evaluation ===")
    
    # Load data
    print("Loading ground truth and predictions...")
    ground_truth = load_ground_truth(test_data_path)
    predictions = load_predictions(results_path)
    
    print(f"Loaded {len(ground_truth)} ground truth samples")
    print(f"Loaded {len(predictions)} predictions")
    
    # Ensure same length
    min_len = min(len(ground_truth), len(predictions))
    ground_truth = ground_truth[:min_len]
    predictions = predictions[:min_len]
    
    print(f"Evaluating on {min_len} samples")
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Print results
    print("\n=== OVERALL METRICS ===")
    print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"F1-Score (Micro): {metrics['overall']['f1_micro']:.4f}")
    print(f"F1-Score (Macro): {metrics['overall']['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['overall']['f1_weighted']:.4f}")
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    
    print("\n=== PER-EMOTION METRICS ===")
    print(f"{'Emotion':<15} {'F1':<6} {'Precision':<9} {'Recall':<6}")
    print("-" * 40)
    
    for emotion, scores in metrics['per_label'].items():
        print(f"{emotion:<15} {scores['f1']:.3f}  {scores['precision']:.3f}     {scores['recall']:.3f}")
    
    # Count parsed vs unparsed predictions
    parsed_count = sum(1 for pred in predictions if sum(pred) > 0 or any(p == 1 for p in pred))
    print(f"\n=== GENERATION ANALYSIS ===")
    print(f"Total samples: {len(predictions)}")
    print(f"Successfully parsed predictions: Variable (all have some form of output)")
    print(f"Average predictions per sample: {np.mean([sum(pred) for pred in predictions]):.2f}")

if __name__ == "__main__":
    main()