#!/usr/bin/env python3
"""
Script to train LFADS on a single 2AFC electrophysiology recording session.

This script configures and runs LFADS training on prepared 2AFC data,
following the pattern of run_single.py.

Author: Assistant
Date: 2024
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import h5py

from lfads_torch.run_model import run_model

def inspect_data_dimensions(data_path):
    """
    Inspect the data file to get dimensions and info.
    
    Parameters:
    -----------
    data_path : str
        Path to the prepared HDF5 data file
    
    Returns:
    --------
    dict : Dictionary with data information
    """
    print(f"Inspecting data file: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Get data dimensions
        train_data_shape = f['train_encod_data'].shape
        n_trials, n_time_bins, n_neurons = train_data_shape
        
        info = {
            'n_trials_train': n_trials,
            'n_trials_valid': f['valid_encod_data'].shape[0],
            'n_trials_test': f['test_encod_data'].shape[0],
            'n_time_bins': n_time_bins,
            'n_neurons': n_neurons,
            'bin_width_ms': f.attrs.get('bin_width_ms', 'unknown'),
            'pre_cue_bins': f.attrs.get('pre_cue_bins', 'unknown'),
            'waiting_period_bins': f.attrs.get('waiting_period_bins', 'unknown'),
            'post_decision_bins': f.attrs.get('post_decision_bins', 'unknown')
        }
        
        print(f"Data dimensions:")
        print(f"  Training trials: {info['n_trials_train']}")
        print(f"  Validation trials: {info['n_trials_valid']}")
        print(f"  Test trials: {info['n_trials_test']}")
        print(f"  Time bins: {info['n_time_bins']}")
        print(f"  Neurons: {info['n_neurons']}")
        print(f"  Bin width: {info['bin_width_ms']} ms")
        print(f"  Pre-cue bins: {info['pre_cue_bins']}")
        print(f"  Waiting period bins: {info['waiting_period_bins']}")
        print(f"  Post-decision bins: {info['post_decision_bins']}")
        
        return info

def main():
    # ---------- OPTIONS -----------
    PROJECT_STR = "2afc-lfads"
    DATASET_STR = "single_session"
    
    # Data and output paths
    # data_path = "G:\\To Process\\PMA17\\PMA17 2020-10-22 Session-1\\Analysis\\LFADS\\PMA17_20201022_Session1.h5"
    # base_output_dir = "G:\\To Process\\PMA17\\PMA17 2020-10-22 Session-1\\Analysis\\LFADS"

    data_path = "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS/PMA17_20201022_Session1.h5"
    base_output_dir = "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS"
    
    # Training parameters
    max_epochs = 200
    factor_dim = 50
    gen_dim = 128
    batch_size = 64
    
    # Create run directory with timestamp
    RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_2afc_single"
    RUN_DIR = Path(base_output_dir) / "runs" / PROJECT_STR / DATASET_STR / RUN_TAG
    OVERWRITE = True
    # ------------------------------

    # Check if data file exists
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Inspect the data
    data_info = inspect_data_dimensions(data_path)
    
    # Overwrite the directory if necessary
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
    
    # Copy this script into the run directory for reproducibility
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    
    # Save data info to run directory
    with open(RUN_DIR / "data_info.txt", "w") as f:
        f.write(f"Data file: {data_path}\n")
        f.write(f"Run started: {datetime.now()}\n\n")
        for key, value in data_info.items():
            f.write(f"{key}: {value}\n")
    
    # Switch to the RUN_DIR
    original_cwd = os.getcwd()
    os.chdir(RUN_DIR)
    
    try:
        print(f"\nStarting LFADS training:")
        print(f"  Working directory: {RUN_DIR}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Factor dim: {factor_dim}")
        print(f"  Generator dim: {gen_dim}")
        print(f"  Batch size: {batch_size}")
        
        # Configuration overrides
        overrides = {
            "datamodule": "2afc_single_session",  # Use your custom datamodule
            "model": "2afc_single_session",  # Use basic model (or create 2afc_single_session model config)
            f"datamodule.datafile_pattern": data_path,
            f"datamodule.batch_size": batch_size,
            # Model dimensions based on actual data
            "model.encod_data_dim": data_info['n_neurons'],
            "model.encod_seq_len": data_info['n_time_bins'],
            "model.recon_seq_len": data_info['n_time_bins'],
            "model.fac_dim": factor_dim,
            "model.gen_dim": gen_dim,
            # Training parameters
            "trainer.max_epochs": max_epochs,
            "seed": 42,
            # Early stopping
            "callbacks.early_stopping.patience": 50,
        }
        
        # Run the model - need to figure out the correct config path
        # Assuming the configs directory is at the same level as scripts
        
        config_path = "../configs/2afc_single.yaml"
        run_model(
            overrides=overrides,
            config_path=str(config_path),
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {RUN_DIR}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        # Return to original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()