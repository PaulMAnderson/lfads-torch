#!/usr/bin/env python3
"""
Script to inspect the contents of an HDF5 data file prepared for LFADS.

This script will show you all the keys, shapes, and attributes in your data file.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path

def inspect_hdf5(file_path):
    """
    Inspect the contents of an HDF5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file to inspect
    """
    print(f"Inspecting HDF5 file: {file_path}")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as f:
        # Print file-level attributes
        print("File attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        print()
        
        # Print all datasets and their properties
        print("Datasets:")
        
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}:")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                
                # Print dataset attributes
                if obj.attrs:
                    print(f"    Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"      {attr_name}: {attr_value}")
                
                # Show some sample data for small datasets or first few values
                if obj.size < 100:
                    print(f"    Data: {obj[...]}")
                elif len(obj.shape) == 1:
                    print(f"    First 5 values: {obj[:5]}")
                elif len(obj.shape) == 2:
                    print(f"    First 3x3: {obj[:3, :3]}")
                elif len(obj.shape) == 3:
                    print(f"    Shape (trials, time, neurons): {obj.shape}")
                    print(f"    Sample from first trial, first 3 time bins: {obj[0, :3, :5]}")
                print()
        
        f.visititems(print_item)
        
        # Show some analysis
        print("Analysis:")
        print("-" * 40)
        
        # Look for standard LFADS keys
        standard_keys = [
            'train_encod_data', 'valid_encod_data', 'test_encod_data',
            'train_recon_data', 'valid_recon_data', 'test_recon_data',
            'train_trial_outcomes', 'valid_trial_outcomes', 'test_trial_outcomes',
            'train_trial_conditions', 'valid_trial_conditions', 'test_trial_conditions',
            'train_trial_types', 'valid_trial_types', 'test_trial_types'
        ]
        
        found_keys = []
        missing_keys = []
        
        for key in standard_keys:
            if key in f:
                found_keys.append(key)
            else:
                missing_keys.append(key)
        
        print(f"Found standard LFADS keys ({len(found_keys)}):")
        for key in found_keys:
            print(f"  ✓ {key}: {f[key].shape}")
        
        print(f"\nMissing standard LFADS keys ({len(missing_keys)}):")
        for key in missing_keys:
            print(f"  ✗ {key}")
        
        # Check for data consistency
        print(f"\nData consistency checks:")
        if 'train_encod_data' in f:
            train_shape = f['train_encod_data'].shape
            print(f"  Training data shape: {train_shape}")
            
            if 'valid_encod_data' in f:
                valid_shape = f['valid_encod_data'].shape
                print(f"  Validation data shape: {valid_shape}")
                
                if train_shape[1:] == valid_shape[1:]:
                    print("  ✓ Train/valid data have consistent dimensions")
                else:
                    print("  ✗ Train/valid data dimension mismatch!")
        
        # Estimate potential batch_keys
        print(f"\nPotential batch_keys for LFADS config:")
        potential_batch_keys = []
        for key in f.keys():
            if key.startswith('train_') and not key.endswith('_data'):
                suffix = key[6:]  # Remove 'train_' prefix
                if f'valid_{suffix}' in f:  # Check if valid version exists
                    potential_batch_keys.append(suffix)
        
        if potential_batch_keys:
            print(f"  {potential_batch_keys}")
        else:
            print("  [] (no additional keys found)")

def main():
    parser = argparse.ArgumentParser(description='Inspect HDF5 data file for LFADS')
    parser.add_argument('data_path', type=str, help='Path to HDF5 data file')
    
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        print(f"Error: File {args.data_path} does not exist!")
        return
    
    try:
        inspect_hdf5(args.data_path)
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # For direct execution, uncomment and modify this path:
    data_path = "G:\\To Process\\PMA17\\PMA17 2020-10-22 Session-1\\Analysis\\LFADS\\PMA17_20201022_Session1.h5"
    # data_path = r"C:\Users\Paul\iCloudDrive\Development\Python\lfads\datasets\mc_maze_large-05ms-val.h5"
    
    import sys
    sys.argv = ['check_data_keys.py', data_path]
    
    main()