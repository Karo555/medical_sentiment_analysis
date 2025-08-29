#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to demonstrate class-weighted training effectiveness.
"""
from __future__ import annotations
import sys
sys.path.append('/workspace/medical_sentiment_analysis')

import torch
import numpy as np
from pathlib import Path
from modules.models.encoder_classifier import EncoderClassifier
from modules.utils.class_weights import load_class_weights, compute_class_weights_from_files
from modules.metrics.classification import compute_all_metrics
from transformers import AutoTokenizer
from modules.data.datasets import EncoderJsonlDataset, EncoderDatasetConfig
from modules.data.collate import encoder_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

def quick_train_test():
    """Quick comparison of class-weighted vs unweighted training."""
    
    print("🔍 TESTING CLASS-WEIGHTED TRAINING")
    print("="*60)
    
    # Load data
    data_dir = Path("data/processed/encoder")
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    # Compute class weights
    print("Computing class weights...")
    weights_result = compute_class_weights_from_files(train_file, method="balanced")
    class_weights = weights_result['pos_weights']
    
    # Setup tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)
    
    # Small dataset for quick testing
    ds_cfg = EncoderDatasetConfig(max_length=128, label_dim=18, return_meta=False)
    train_ds = EncoderJsonlDataset(train_file, tokenizer, cfg=ds_cfg)
    val_ds = EncoderJsonlDataset(val_file, tokenizer, cfg=ds_cfg)
    
    collate_fn = encoder_collate(tokenizer, pad_to_multiple_of=8)
    
    # Create small dataloaders for quick test
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Test both models
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_type, weights in [("unweighted", None), ("weighted", class_weights)]:
        print(f"\n--- Testing {model_type.upper()} model ---")
        
        # Create model
        model = EncoderClassifier(
            "xlm-roberta-base", 
            out_dim=18, 
            class_weights=weights,
            use_binary_classification=True
        )
        model.to(device)
        
        # Quick training (1 epoch, few batches)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        
        total_loss = 0
        num_batches = 0
        max_batches = 10  # Quick test
        
        print(f"Training {model_type} model (max {max_batches} batches)...")
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            if i >= max_batches:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Quick evaluation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Evaluation")):
                if i >= 5:  # Quick eval
                    break
                    
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                # Get probabilities and predictions
                logits = outputs["logits"]
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
                
                val_loss += outputs["loss"].item()
                val_batches += 1
        
        # Compute metrics
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        metrics = compute_all_metrics(y_true, y_pred)
        
        results[model_type] = {
            'train_loss': avg_train_loss,
            'val_loss': val_loss / val_batches,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_true': y_true
        }
        
        print(f"Validation loss: {val_loss / val_batches:.4f}")
        print(f"Validation metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    unweighted = results['unweighted']
    weighted = results['weighted']
    
    print(f"Training Loss:")
    print(f"  Unweighted: {unweighted['train_loss']:.4f}")
    print(f"  Weighted:   {weighted['train_loss']:.4f}")
    print(f"  Difference: {weighted['train_loss'] - unweighted['train_loss']:.4f}")
    
    print(f"\nValidation Metrics:")
    metrics_to_compare = ['accuracy', 'f1_score', 'precision', 'recall']
    
    for metric in metrics_to_compare:
        unw_val = unweighted['metrics'][metric]
        w_val = weighted['metrics'][metric]
        improvement = ((w_val - unw_val) / unw_val) * 100 if unw_val > 0 else 0
        
        print(f"  {metric.capitalize()}:")
        print(f"    Unweighted: {unw_val:.4f}")
        print(f"    Weighted:   {w_val:.4f}")
        print(f"    Change:     {improvement:+.1f}%")
    
    # Check prediction patterns
    print(f"\nPrediction Analysis:")
    
    unw_pred_counts = np.sum(results['unweighted']['y_pred'], axis=0)
    w_pred_counts = np.sum(results['weighted']['y_pred'], axis=0)
    true_counts = np.sum(results['unweighted']['y_true'], axis=0)
    
    print(f"  Total positive predictions:")
    print(f"    Unweighted: {np.sum(unw_pred_counts):.0f}")
    print(f"    Weighted:   {np.sum(w_pred_counts):.0f}")
    print(f"    True:       {np.sum(true_counts):.0f}")
    
    # Check most imbalanced labels
    print(f"\nMost imbalanced labels (with highest class weights):")
    sorted_indices = torch.argsort(class_weights, descending=True)[:5]
    
    for i, idx in enumerate(sorted_indices):
        idx = idx.item()
        weight = class_weights[idx].item()
        true_count = true_counts[idx]
        unw_pred = unw_pred_counts[idx]
        w_pred = w_pred_counts[idx]
        
        print(f"  Label {idx} (weight={weight:.1f}):")
        print(f"    True: {true_count}, Unweighted pred: {unw_pred}, Weighted pred: {w_pred}")
    
    print(f"\n✅ Class-weighted training test completed!")
    return results

if __name__ == "__main__":
    results = quick_train_test()