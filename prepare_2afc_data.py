#!/usr/bin/env python3
"""
Data preparation script for 2AFC electrophysiology data for LFADS analysis.

This script extracts and formats spike data from a 2 Alternative Forced Choice task
with variable waiting periods following auditory cues. The focus is on analyzing
dynamics during the waiting period, particularly comparing:
- Trials where the animal is rewarded (correct choices)
- Error trials (incorrect choices) 
- Trials where the animal gives up before reward

Author: Assistant
Date: 2024
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings

class TwoAFCDataProcessor:
    """
    Processor for 2AFC electrophysiology data to prepare for LFADS analysis.
    """
    
    def __init__(self, 
                 bin_width_ms: float = 20.0,
                 smoothing_std_ms: float = 20.0,
                 pre_cue_ms: float = 500.0,
                 waiting_period_ms: float = 2000.0,
                 post_decision_ms: float = 500.0):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        bin_width_ms : float
            Bin width for spike binning in milliseconds
        smoothing_std_ms : float
            Standard deviation for Gaussian smoothing kernel in milliseconds
        pre_cue_ms : float
            Time before cue onset to include in analysis
        waiting_period_ms : float
            Maximum waiting period duration to analyze
        post_decision_ms : float
            Time after decision/outcome to include
        """
        self.bin_width_ms = bin_width_ms
        self.bin_width_s = bin_width_ms / 1000.0
        self.smoothing_std_ms = smoothing_std_ms
        self.pre_cue_ms = pre_cue_ms
        self.waiting_period_ms = waiting_period_ms
        self.post_decision_ms = post_decision_ms
        
        # Calculate total analysis window
        self.total_time_ms = pre_cue_ms + waiting_period_ms + post_decision_ms
        self.n_time_bins = int(self.total_time_ms / bin_width_ms)
        
        print(f"Data processor initialized:")
        print(f"  Bin width: {bin_width_ms} ms")
        print(f"  Analysis window: {self.total_time_ms} ms ({self.n_time_bins} bins)")
        print(f"  Pre-cue: {pre_cue_ms} ms")
        print(f"  Waiting period: {waiting_period_ms} ms") 
        print(f"  Post-decision: {post_decision_ms} ms")
    
    def load_spike_data(self, data_path: str) -> Dict:
        """
        Load spike data from your recording format.
        
        This is a template function - you'll need to modify this based on
        your actual data format (e.g., .mat files, .npy, .pkl, etc.)
        
        Expected output structure:
        {
            'spike_times': List of arrays, one per neuron containing spike times
            'trial_info': Dict with trial metadata
            'neuron_info': Dict with neuron metadata  
        }
        """
        # TODO: Implement based on your actual data format
        # This is a placeholder - replace with your actual loading code
        
        print(f"Loading data from: {data_path}")
        
        # Example structure for different file types:
        if data_path.endswith('.mat'):
            from scipy.io import loadmat
            data = loadmat(data_path)
            # Extract your spike times, trial info, etc.
            
        elif data_path.endswith('.npy'):
            data = np.load(data_path, allow_pickle=True).item()
            
        elif data_path.endswith('.pkl'):
            import pickle
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Return standardized format
        return {
            'spike_times': data.get('spike_times', []),
            'trial_info': data.get('trial_info', {}),
            'neuron_info': data.get('neuron_info', {})
        }
    
    def extract_trial_epochs(self, trial_info: Dict) -> Dict:
        """
        Extract relevant time epochs for each trial.
        
        Parameters:
        -----------
        trial_info : dict
            Trial information containing timestamps and outcomes
            
        Expected keys in trial_info:
        - 'cue_times': Array of cue onset times
        - 'decision_times': Array of decision times (when animal responds)
        - 'outcome_times': Array of outcome times (reward/error/timeout)
        - 'trial_outcomes': Array of trial outcomes (0=error, 1=correct, 2=timeout/giveup)
        - 'evidence_levels': Array of evidence levels for each trial
        - 'choices': Array of animal choices (0=left, 1=right)
        
        Returns:
        --------
        dict: Trial epochs with start/end times for analysis windows
        """
        
        cue_times = trial_info['cue_times']
        decision_times = trial_info.get('decision_times', cue_times + self.waiting_period_ms/1000)
        outcome_times = trial_info.get('outcome_times', decision_times)
        
        n_trials = len(cue_times)
        
        # Define analysis window for each trial
        trial_epochs = {
            'start_times': cue_times - self.pre_cue_ms/1000,
            'end_times': cue_times + (self.waiting_period_ms + self.post_decision_ms)/1000,
            'cue_times': cue_times,
            'decision_times': decision_times,
            'outcome_times': outcome_times,
            'trial_outcomes': trial_info.get('trial_outcomes', np.zeros(n_trials)),
            'evidence_levels': trial_info.get('evidence_levels', np.zeros(n_trials)),
            'choices': trial_info.get('choices', np.zeros(n_trials))
        }
        
        return trial_epochs
    
    def bin_spikes(self, spike_times: List[np.ndarray], trial_epochs: Dict) -> np.ndarray:
        """
        Bin spike times into trial-aligned arrays.
        
        Parameters:
        -----------
        spike_times : list
            List of arrays, one per neuron containing spike times
        trial_epochs : dict
            Trial timing information
            
        Returns:
        --------
        np.ndarray: Binned spikes of shape (n_trials, n_time_bins, n_neurons)
        """
        n_neurons = len(spike_times)
        n_trials = len(trial_epochs['start_times'])
        
        # Initialize binned spike array
        binned_spikes = np.zeros((n_trials, self.n_time_bins, n_neurons))
        
        for trial_idx in range(n_trials):
            start_time = trial_epochs['start_times'][trial_idx]
            end_time = trial_epochs['end_times'][trial_idx]
            
            # Create time bins for this trial
            time_bins = np.linspace(start_time, end_time, self.n_time_bins + 1)
            
            for neuron_idx, neuron_spikes in enumerate(spike_times):
                # Find spikes within trial window
                trial_spikes = neuron_spikes[
                    (neuron_spikes >= start_time) & (neuron_spikes < end_time)
                ]
                
                # Bin the spikes
                spike_counts, _ = np.histogram(trial_spikes, bins=time_bins)
                binned_spikes[trial_idx, :, neuron_idx] = spike_counts
        
        # Convert to firing rates (spikes/second)
        binned_spikes = binned_spikes / self.bin_width_s
        
        return binned_spikes
    
    def smooth_spikes(self, binned_spikes: np.ndarray) -> np.ndarray:
        """
        Smooth binned spikes with Gaussian kernel.
        
        Parameters:
        -----------
        binned_spikes : np.ndarray
            Binned spike data of shape (n_trials, n_time_bins, n_neurons)
            
        Returns:
        --------
        np.ndarray: Smoothed spike data
        """
        # Create Gaussian smoothing kernel
        std_bins = self.smoothing_std_ms / self.bin_width_ms
        window_length = int(std_bins * 6)  # 3 std on each side
        if window_length % 2 == 0:
            window_length += 1  # Make odd
        
        window = gaussian(window_length, std_bins)
        window = window / window.sum()  # Normalize
        
        # Apply smoothing
        n_trials, n_time_bins, n_neurons = binned_spikes.shape
        smoothed_spikes = np.zeros_like(binned_spikes)
        
        # Remove convolution artifacts
        pad_length = window_length // 2
        
        for trial_idx in range(n_trials):
            for neuron_idx in range(n_neurons):
                # Apply convolution with padding
                padded_signal = np.pad(
                    binned_spikes[trial_idx, :, neuron_idx], 
                    pad_length, 
                    mode='edge'
                )
                smoothed_signal = lfilter(window, 1, padded_signal)
                smoothed_spikes[trial_idx, :, neuron_idx] = smoothed_signal[pad_length:-pad_length]
        
        return smoothed_spikes
    
    def categorize_trials(self, trial_epochs: Dict) -> Dict[str, np.ndarray]:
        """
        Categorize trials based on outcome and behavior.
        
        Parameters:
        -----------
        trial_epochs : dict
            Trial information including outcomes
            
        Returns:
        --------
        dict: Trial indices for different categories
        """
        outcomes = trial_epochs['trial_outcomes']
        
        # Define trial categories
        categories = {
            'correct': outcomes == 1,
            'error': outcomes == 0, 
            'timeout_giveup': outcomes == 2,
            'all_completed': (outcomes == 0) | (outcomes == 1),  # Exclude timeouts
            'all_trials': np.ones(len(outcomes), dtype=bool)
        }
        
        # Add evidence level categories if available
        if 'evidence_levels' in trial_epochs:
            evidence = trial_epochs['evidence_levels']
            unique_evidence = np.unique(evidence)
            for ev in unique_evidence:
                categories[f'evidence_{ev}'] = evidence == ev
        
        # Print trial counts
        print("\nTrial categorization:")
        for category, mask in categories.items():
            print(f"  {category}: {np.sum(mask)} trials")
        
        return categories
    
    def create_lfads_dataset(self, 
                           smoothed_spikes: np.ndarray,
                           trial_categories: Dict[str, np.ndarray],
                           trial_epochs: Dict,
                           focus_categories: List[str] = ['correct', 'error'],
                           train_ratio: float = 0.8,
                           random_state: int = 42) -> Dict:
        """
        Create dataset in LFADS format.
        
        Parameters:
        -----------
        smoothed_spikes : np.ndarray
            Smoothed spike data (n_trials, n_time_bins, n_neurons)
        trial_categories : dict
            Trial category masks
        trial_epochs : dict
            Trial timing information
        focus_categories : list
            Categories to include in the analysis
        train_ratio : float
            Fraction of data for training
        random_state : int
            Random seed for train/test split
            
        Returns:
        --------
        dict: LFADS-formatted dataset
        """
        
        # Select trials from focus categories
        if focus_categories:
            focus_mask = np.zeros(smoothed_spikes.shape[0], dtype=bool)
            for category in focus_categories:
                if category in trial_categories:
                    focus_mask |= trial_categories[category]
            
            focus_data = smoothed_spikes[focus_mask]
            focus_info = {key: val[focus_mask] for key, val in trial_epochs.items() 
                         if isinstance(val, np.ndarray) and len(val) == smoothed_spikes.shape[0]}
        else:
            focus_data = smoothed_spikes
            focus_info = trial_epochs
        
        n_trials, n_time_bins, n_neurons = focus_data.shape
        
        print(f"\nCreating LFADS dataset:")
        print(f"  Selected {n_trials} trials from categories: {focus_categories}")
        print(f"  Data shape: {focus_data.shape}")
        
        # Train/validation split
        train_indices, valid_indices = train_test_split(
            np.arange(n_trials),
            train_size=train_ratio,
            random_state=random_state,
            stratify=focus_info.get('trial_outcomes')  # Stratify by outcome if available
        )
        
        # Create LFADS dataset
        dataset = {
            'train_encod_data': focus_data[train_indices].astype(np.float32),
            'train_recon_data': focus_data[train_indices].astype(np.float32),
            'valid_encod_data': focus_data[valid_indices].astype(np.float32), 
            'valid_recon_data': focus_data[valid_indices].astype(np.float32),
            'train_indices': train_indices.astype(np.int32),
            'valid_indices': valid_indices.astype(np.int32),
            # Additional metadata
            'trial_outcomes_train': focus_info['trial_outcomes'][train_indices].astype(np.int32),
            'trial_outcomes_valid': focus_info['trial_outcomes'][valid_indices].astype(np.int32),
            'evidence_levels_train': focus_info.get('evidence_levels', np.zeros(len(train_indices)))[train_indices].astype(np.float32),
            'evidence_levels_valid': focus_info.get('evidence_levels', np.zeros(len(valid_indices)))[valid_indices].astype(np.float32),
            'choices_train': focus_info.get('choices', np.zeros(len(train_indices)))[train_indices].astype(np.int32),
            'choices_valid': focus_info.get('choices', np.zeros(len(valid_indices)))[valid_indices].astype(np.int32),
            # Timing information
            'bin_width_ms': self.bin_width_ms,
            'pre_cue_bins': int(self.pre_cue_ms / self.bin_width_ms),
            'waiting_period_bins': int(self.waiting_period_ms / self.bin_width_ms),
            'post_decision_bins': int(self.post_decision_ms / self.bin_width_ms)
        }
        
        print(f"  Training trials: {len(train_indices)}")
        print(f"  Validation trials: {len(valid_indices)}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict, output_path: str):
        """
        Save dataset to HDF5 file.
        
        Parameters:
        -----------
        dataset : dict
            LFADS-formatted dataset
        output_path : str
            Output file path
        """
        print(f"\nSaving dataset to: {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to HDF5
        with h5py.File(output_path, 'w') as f:
            for key, value in dataset.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                else:
                    # Save scalars as attributes
                    f.attrs[key] = value
        
        print(f"Dataset saved successfully!")
        
        # Print summary
        with h5py.File(output_path, 'r') as f:
            print("\nDataset contents:")
            for key in f.keys():
                if hasattr(f[key], 'shape'):
                    print(f"  {key}: {f[key].shape}")
                    
    def plot_data_summary(self, 
                         smoothed_spikes: np.ndarray,
                         trial_categories: Dict[str, np.ndarray],
                         trial_epochs: Dict,
                         output_dir: str = None):
        """
        Create summary plots of the data.
        
        Parameters:
        -----------
        smoothed_spikes : np.ndarray
            Smoothed spike data
        trial_categories : dict
            Trial category masks
        trial_epochs : dict
            Trial timing information
        output_dir : str, optional
            Directory to save plots
        """
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Time axis in seconds
        time_axis = np.linspace(
            -self.pre_cue_ms/1000, 
            (self.waiting_period_ms + self.post_decision_ms)/1000,
            self.n_time_bins
        )
        
        # Plot 1: Average firing rates by trial type
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        categories_to_plot = ['correct', 'error', 'timeout_giveup']
        colors = ['green', 'red', 'orange']
        
        # Population average
        ax = axes[0, 0]
        for category, color in zip(categories_to_plot, colors):
            if category in trial_categories and np.sum(trial_categories[category]) > 0:
                category_data = smoothed_spikes[trial_categories[category]]
                pop_avg = np.mean(category_data, axis=(0, 2))  # Average across trials and neurons
                ax.plot(time_axis, pop_avg, color=color, label=f'{category} (n={np.sum(trial_categories[category])})')
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cue onset')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Population firing rate (Hz)')
        ax.set_title('Population Average Firing Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Example neuron responses
        ax = axes[0, 1]
        neuron_idx = 0  # Plot first neuron
        for category, color in zip(categories_to_plot, colors):
            if category in trial_categories and np.sum(trial_categories[category]) > 0:
                category_data = smoothed_spikes[trial_categories[category]]
                neuron_avg = np.mean(category_data[:, :, neuron_idx], axis=0)
                ax.plot(time_axis, neuron_avg, color=color, label=f'{category}')
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cue onset')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'Example Neuron {neuron_idx} Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Raster plot for one trial type
        ax = axes[1, 0]
        if 'correct' in trial_categories:
            correct_trials = smoothed_spikes[trial_categories['correct']]
            if len(correct_trials) > 0:
                # Show first 20 trials, all neurons
                n_show_trials = min(20, len(correct_trials))
                raster_data = correct_trials[:n_show_trials, :, :].reshape(n_show_trials * self.n_time_bins, -1)
                im = ax.imshow(raster_data.T, aspect='auto', cmap='viridis', 
                              extent=[0, n_show_trials, 0, smoothed_spikes.shape[2]])
                ax.set_xlabel('Trial')
                ax.set_ylabel('Neuron')
                ax.set_title('Firing Rates - Correct Trials')
                plt.colorbar(im, ax=ax, label='Firing rate (Hz)')
        
        # Trial outcome distribution
        ax = axes[1, 1]
        outcomes = trial_epochs['trial_outcomes']
        unique_outcomes, counts = np.unique(outcomes, return_counts=True)
        outcome_labels = ['Error', 'Correct', 'Timeout/Giveup']
        ax.bar(range(len(unique_outcomes)), counts, 
               color=['red', 'green', 'orange'][:len(unique_outcomes)])
        ax.set_xticks(range(len(unique_outcomes)))
        ax.set_xticklabels([outcome_labels[i] for i in unique_outcomes])
        ax.set_ylabel('Number of trials')
        ax.set_title('Trial Outcome Distribution')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'data_summary.png'), dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to: {os.path.join(output_dir, 'data_summary.png')}")
        
        plt.show()


def main():
    """
    Main function to process 2AFC data for LFADS.
    """
    parser = argparse.ArgumentParser(description='Prepare 2AFC electrophysiology data for LFADS')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path for output HDF5 file')
    parser.add_argument('--bin_width_ms', type=float, default=20.0,
                       help='Bin width in milliseconds')
    parser.add_argument('--smoothing_std_ms', type=float, default=20.0,
                       help='Gaussian smoothing standard deviation in milliseconds')
    parser.add_argument('--pre_cue_ms', type=float, default=500.0,
                       help='Time before cue onset to include (ms)')
    parser.add_argument('--waiting_period_ms', type=float, default=2000.0,
                       help='Waiting period duration to analyze (ms)')
    parser.add_argument('--post_decision_ms', type=float, default=500.0,
                       help='Time after decision to include (ms)')
    parser.add_argument('--focus_categories', nargs='+', 
                       default=['correct', 'error'],
                       help='Trial categories to include in analysis')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--plot_summary', action='store_true',
                       help='Generate summary plots')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = TwoAFCDataProcessor(
        bin_width_ms=args.bin_width_ms,
        smoothing_std_ms=args.smoothing_std_ms,
        pre_cue_ms=args.pre_cue_ms,
        waiting_period_ms=args.waiting_period_ms,
        post_decision_ms=args.post_decision_ms
    )
    
    # Load and process data
    print("Loading spike data...")
    raw_data = processor.load_spike_data(args.data_path)
    
    print("Extracting trial epochs...")
    trial_epochs = processor.extract_trial_epochs(raw_data['trial_info'])
    
    print("Binning spikes...")
    binned_spikes = processor.bin_spikes(raw_data['spike_times'], trial_epochs)
    
    print("Smoothing spikes...")
    smoothed_spikes = processor.smooth_spikes(binned_spikes)
    
    print("Categorizing trials...")
    trial_categories = processor.categorize_trials(trial_epochs)
    
    print("Creating LFADS dataset...")
    dataset = processor.create_lfads_dataset(
        smoothed_spikes, 
        trial_categories, 
        trial_epochs,
        focus_categories=args.focus_categories,
        train_ratio=args.train_ratio
    )
    
    # Save dataset
    processor.save_dataset(dataset, args.output_path)
    
    # Generate plots if requested
    if args.plot_summary:
        output_dir = os.path.dirname(args.output_path)
        processor.plot_data_summary(smoothed_spikes, trial_categories, trial_epochs, output_dir)
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()