#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove underrepresented labels from the medical sentiment analysis dataset.
This script removes 'embarrassing', 'vulgar', and 'more' labels from all datasets
and updates the schema to reflect the new 18-dimensional label space.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import shutil
import numpy as np

# Labels to remove based on severe underrepresentation
LABELS_TO_REMOVE = ["embarassing", "vulgar", "more"]  # Note: "embarrassing" is misspelled in original
REMOVE_INDICES = [13, 14, 18]  # 0-based indices in the original 21-label schema

def load_label_names(schema_path: Path) -> List[str]:
    """Load the current label names."""
    with schema_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def create_new_label_schema(old_labels: List[str], remove_indices: List[int]) -> List[str]:
    """Create new label schema with specified indices removed."""
    new_labels = []
    for i, label in enumerate(old_labels):
        if i not in remove_indices:
            new_labels.append(label)
    return new_labels

def remove_labels_from_array(labels_array: List[int], remove_indices: List[int]) -> List[int]:
    """Remove specified indices from a label array."""
    return [labels_array[i] for i in range(len(labels_array)) if i not in remove_indices]

def process_dataset_file(input_file: Path, output_file: Path, remove_indices: List[int]) -> Dict[str, int]:
    """Process a single JSONL dataset file to remove specified labels."""
    
    stats = {
        'total_samples': 0,
        'processed_samples': 0,
        'removed_positive_labels': {idx: 0 for idx in remove_indices}
    }
    
    print(f"Processing {input_file} -> {output_file}")
    
    with input_file.open('r', encoding='utf-8') as infile, \
         output_file.open('w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                stats['total_samples'] += 1
                
                # Check original labels
                original_labels = data['labels']
                if len(original_labels) != 21:
                    print(f"Warning: Sample {line_num} has {len(original_labels)} labels, expected 21")
                
                # Count positive labels being removed
                for idx in remove_indices:
                    if idx < len(original_labels) and original_labels[idx] == 1:
                        stats['removed_positive_labels'][idx] += 1
                
                # Remove specified labels
                new_labels = remove_labels_from_array(original_labels, remove_indices)
                
                # Update the data
                data['labels'] = new_labels
                
                # Write processed sample
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                stats['processed_samples'] += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    return stats

def update_configs_for_new_dimensions(config_dir: Path, old_dim: int, new_dim: int):
    """Update configuration files to reflect new label dimensions."""
    
    print(f"\nUpdating configuration files: {old_dim}D -> {new_dim}D")
    
    # Update all encoder experiment configs
    encoder_configs = list((config_dir / "experiment").glob("enc_*.yaml"))
    
    for config_file in encoder_configs:
        print(f"Updating {config_file}")
        
        # Read config
        with config_file.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace dimension references
        content = content.replace(f'out_dim: {old_dim}', f'out_dim: {new_dim}')
        content = content.replace(f'label_dim: {old_dim}', f'label_dim: {new_dim}')
        
        # Write back
        with config_file.open('w', encoding='utf-8') as f:
            f.write(content)

def create_label_mapping_file(old_labels: List[str], new_labels: List[str], 
                             remove_indices: List[int], output_path: Path):
    """Create a mapping file documenting the label transformation."""
    
    mapping = {
        'transformation': f'{len(old_labels)}D -> {len(new_labels)}D',
        'removed_labels': {
            'names': [old_labels[i] for i in remove_indices],
            'original_indices': remove_indices,
            'reason': 'Severely underrepresented in dataset'
        },
        'label_mapping': {
            'old_to_new': {},
            'new_to_old': {}
        },
        'old_schema': old_labels,
        'new_schema': new_labels
    }
    
    # Create index mappings
    new_idx = 0
    for old_idx, label in enumerate(old_labels):
        if old_idx not in remove_indices:
            mapping['label_mapping']['old_to_new'][old_idx] = new_idx
            mapping['label_mapping']['new_to_old'][new_idx] = old_idx
            new_idx += 1
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Label mapping saved to: {output_path}")

def main():
    """Main function to remove underrepresented labels from the pipeline."""
    
    project_root = Path("/workspace/medical_sentiment_analysis")
    
    print("🗑️  REMOVING UNDERREPRESENTED LABELS FROM PIPELINE")
    print("=" * 60)
    print(f"Labels to remove: {LABELS_TO_REMOVE}")
    print(f"Original indices: {REMOVE_INDICES}")
    
    # Load current schema
    schema_path = project_root / "schema" / "label_names.json"
    old_labels = load_label_names(schema_path)
    print(f"Original schema: {len(old_labels)} labels")
    
    # Verify indices match label names
    for idx, label_name in zip(REMOVE_INDICES, LABELS_TO_REMOVE):
        if old_labels[idx] != label_name:
            print(f"ERROR: Index {idx} is '{old_labels[idx]}', expected '{label_name}'")
            return
    
    # Create new schema
    new_labels = create_new_label_schema(old_labels, REMOVE_INDICES)
    print(f"New schema: {len(new_labels)} labels")
    
    # Backup and update schema
    backup_schema = schema_path.parent / "label_names_21d_backup.json"
    shutil.copy2(schema_path, backup_schema)
    print(f"Backed up original schema to: {backup_schema}")
    
    with schema_path.open('w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=2, ensure_ascii=False)
    print(f"Updated schema saved to: {schema_path}")
    
    # Process datasets
    data_dir = project_root / "data" / "processed" / "encoder"
    backup_dir = project_root / "data" / "processed" / "encoder_21d_backup"
    
    # Create backup
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(data_dir, backup_dir)
    print(f"Backed up original datasets to: {backup_dir}")
    
    # Process each split
    all_stats = {}
    for split in ["train", "val", "test"]:
        input_file = data_dir / f"{split}.jsonl"
        output_file = data_dir / f"{split}_18d.jsonl"
        
        if input_file.exists():
            stats = process_dataset_file(input_file, output_file, REMOVE_INDICES)
            all_stats[split] = stats
            
            # Replace original with processed
            input_file.unlink()
            output_file.rename(input_file)
            
        else:
            print(f"Warning: {input_file} not found")
    
    # Print processing statistics
    print(f"\n📊 PROCESSING STATISTICS")
    print("=" * 40)
    
    total_removed = {idx: 0 for idx in REMOVE_INDICES}
    for split, stats in all_stats.items():
        print(f"\n{split.upper()} split:")
        print(f"  Samples processed: {stats['processed_samples']}")
        
        for idx in REMOVE_INDICES:
            removed = stats['removed_positive_labels'][idx]
            total_removed[idx] += removed
            label_name = old_labels[idx]
            print(f"  Removed positive '{label_name}' labels: {removed}")
    
    print(f"\nTOTAL POSITIVE LABELS REMOVED:")
    for idx in REMOVE_INDICES:
        label_name = old_labels[idx]
        print(f"  {label_name}: {total_removed[idx]} positive samples")
    
    # Update configuration files
    config_dir = project_root / "configs"
    update_configs_for_new_dimensions(config_dir, 21, 18)
    
    # Create label mapping documentation
    mapping_path = project_root / "schema" / "label_dimension_reduction.json"
    create_label_mapping_file(old_labels, new_labels, REMOVE_INDICES, mapping_path)
    
    # Summary
    print(f"\n✅ LABEL REMOVAL COMPLETED")
    print("=" * 40)
    print(f"✓ Schema updated: 21D -> 18D")
    print(f"✓ Datasets processed: {len(all_stats)} splits")
    print(f"✓ Configs updated: dimension references")
    print(f"✓ Backups created for rollback")
    print(f"✓ Mapping documented: {mapping_path}")
    
    print(f"\nRemoved labels:")
    for idx, label_name in zip(REMOVE_INDICES, LABELS_TO_REMOVE):
        print(f"  - {label_name} (was index {idx})")
    
    print(f"\nNext steps:")
    print("1. Update model architectures to use out_dim=18")
    print("2. Retrain models with new 18D schema") 
    print("3. Re-run class imbalance analysis")
    
    return all_stats

if __name__ == "__main__":
    main()