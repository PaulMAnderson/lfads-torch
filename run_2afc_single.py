#!/usr/bin/env python3
"""
Script to train LFADS on a single 2AFC electrophysiology recording session.

This script configures and runs LFADS training on prepared 2AFC data,
focusing on dynamics during the waiting period.

Author: Assistant
Date: 2024
"""

import os
import sys
from pathlib import Path
import argparse
import yaml

# Add lfads_torch to path if needed
sys.path.append(str(Path(__file__).parent))

from lfads_torch.run_model import run_model

def update_config_for_data(data_path: str, config_overrides: dict):
    """
    Update model configuration based on actual data dimensions.
    
    Parameters:
    -----------
    data_path : str
        Path to the prepared HDF5 data file
    config_overrides : dict
        Configuration overrides to update
    """
    import h5py
    
    print(f"Reading data dimensions from: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Get data dimensions
        train_data_shape = f['train_encod_data'].shape
        n_trials, n_time_bins, n_neurons = train_data_shape
        
        print(f"Data dimensions:")
        print(f"  Trials: {n_trials}")
        print(f"  Time bins: {n_time_bins}")
        print(f"  Neurons: {n_neurons}")
        
        # Update model configuration
        config_overrides['model.encod_data_dim'] = n_neurons
        config_overrides['model.encod_seq_len'] = n_time_bins
        config_overrides['model.recon_seq_len'] = n_time_bins
        config_overrides['model.readout.modules.0.out_features'] = n_neurons
        
        # Get timing information
        if 'bin_width_ms' in f.attrs:
            bin_width = f.attrs['bin_width_ms']
            print(f"  Bin width: {bin_width} ms")
        
        if 'pre_cue_bins' in f.attrs:
            pre_cue_bins = f.attrs['pre_cue_bins']
            waiting_bins = f.attrs.get('waiting_period_bins', 0)
            post_bins = f.attrs.get('post_decision_bins', 0)
            print(f"  Pre-cue: {pre_cue_bins} bins")
            print(f"  Waiting period: {waiting_bins} bins") 
            print(f"  Post-decision: {post_bins} bins")
    
    return config_overrides

def main():
    """
    Main function to run LFADS on 2AFC data.
    """
    parser = argparse.ArgumentParser(description='Train LFADS on 2AFC electrophysiology data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared HDF5 data file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for model outputs')
    parser.add_argument('--config_name', type=str, default='2afc_single_session',
                       help='Name of the configuration to use')
    parser.add_argument('--max_epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--factor_dim', type=int, default=50,
                       help='Latent factor dimensionality')
    parser.add_argument('--gen_dim', type=int, default=128,
                       help='Generator RNN dimensionality')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Base configuration overrides
    config_overrides = {
        'datamodule': '2afc_single_session',
        'model': '2afc_single_session',
        f'datamodule.datafile_pattern': args.data_path,
        f'datamodule.batch_size': args.batch_size,
        'model.fac_dim': args.factor_dim,
        'model.gen_dim': args.gen_dim,
        'trainer.max_epochs': args.max_epochs,
        'trainer.gpus': args.gpus if args.gpus > 0 else None,
        'seed': args.seed,
        'callbacks.model_checkpoint.patience': args.patience,
        'callbacks.early_stopping.patience': args.patience,
    }
    
    # Update config based on actual data dimensions
    config_overrides = update_config_for_data(args.data_path, config_overrides)
    
    print(f"\nStarting LFADS training:")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Factor dim: {args.factor_dim}")
    print(f"  Generator dim: {args.gen_dim}")
    print(f"  Batch size: {args.batch_size}")
    
    # Run the model
    try:
        run_model(
            config_path=f'configs/{args.config_name}.yaml',
            overrides=config_overrides
        )
        print(f"\nTraining completed successfully!")
        print(f"Model outputs saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

if __name__ == "__main__":
    main()