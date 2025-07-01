#!/usr/bin/env python3
"""
Example script showing how to adapt the data loading function for your specific data format.

This script provides templates for different common electrophysiology data formats
and shows how to structure your trial information for the 2AFC analysis.

Author: Assistant
Date: 2024
"""

import numpy as np
from typing import Dict, List, Tuple
import pickle
from scipy.io import loadmat

def load_spike_data_example_matlab(data_path: str) -> Dict:
    """
    Example data loading for MATLAB .mat files.
    
    Adapt this function based on your actual MATLAB file structure.
    
    Expected MATLAB structure:
    - spike_times: cell array, each cell contains spike times for one neuron
    - trial_data: structure with trial information
    """
    
    print(f"Loading MATLAB data from: {data_path}")
    
    # Load MATLAB file
    mat_data = loadmat(data_path)
    
    # Extract spike times (assuming cell array format)
    spike_times_cell = mat_data['spike_times']  # Cell array
    spike_times = []
    
    for i in range(spike_times_cell.shape[0]):
        neuron_spikes = spike_times_cell[i, 0].flatten()
        spike_times.append(neuron_spikes)
    
    # Extract trial information
    trial_data = mat_data['trial_data']  # Structure
    
    trial_info = {
        'cue_times': trial_data['cue_onset'][0, 0].flatten(),
        'decision_times': trial_data['response_time'][0, 0].flatten(),
        'outcome_times': trial_data['outcome_time'][0, 0].flatten(),
        'trial_outcomes': trial_data['outcome'][0, 0].flatten(),  # 0=error, 1=correct, 2=timeout
        'evidence_levels': trial_data['evidence_level'][0, 0].flatten(),
        'choices': trial_data['choice'][0, 0].flatten(),  # 0=left, 1=right
        'reaction_times': trial_data['reaction_time'][0, 0].flatten()
    }
    
    # Extract neuron information if available
    neuron_info = {
        'neuron_ids': mat_data.get('neuron_ids', np.arange(len(spike_times))),
        'brain_area': mat_data.get('brain_area', ['unknown'] * len(spike_times)),
        'electrode_info': mat_data.get('electrode_info', {})
    }
    
    return {
        'spike_times': spike_times,
        'trial_info': trial_info,
        'neuron_info': neuron_info
    }

def load_spike_data_example_python(data_path: str) -> Dict:
    """
    Example data loading for Python pickle files.
    
    Adapt this function based on your actual pickle file structure.
    """
    
    print(f"Loading Python data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Assuming your pickle file has a dictionary structure
    spike_times = data['spike_times']  # List of arrays
    
    trial_info = {
        'cue_times': data['trials']['cue_onset'],
        'decision_times': data['trials']['response_time'],
        'outcome_times': data['trials']['outcome_time'],
        'trial_outcomes': data['trials']['outcome'],
        'evidence_levels': data['trials']['evidence_level'],
        'choices': data['trials']['choice'],
        'reaction_times': data['trials']['reaction_time']
    }
    
    neuron_info = data.get('neurons', {})
    
    return {
        'spike_times': spike_times,
        'trial_info': trial_info,
        'neuron_info': neuron_info
    }

def load_spike_data_example_nwb(data_path: str) -> Dict:
    """
    Example data loading for Neurodata Without Borders (NWB) files.
    
    Requires pynwb: pip install pynwb
    """
    
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        raise ImportError("pynwb required for NWB files. Install with: pip install pynwb")
    
    print(f"Loading NWB data from: {data_path}")
    
    with NWBHDF5IO(data_path, 'r') as io:
        nwbfile = io.read()
        
        # Extract spike times
        spike_times = []
        units = nwbfile.units
        
        for unit_id in units.id[:]:
            unit_spike_times = units.get_unit_spike_times(unit_id)
            spike_times.append(unit_spike_times)
        
        # Extract trial information
        trials = nwbfile.trials
        
        trial_info = {
            'cue_times': trials['cue_onset'][:],
            'decision_times': trials['response_time'][:],
            'outcome_times': trials['outcome_time'][:],
            'trial_outcomes': trials['outcome'][:],
            'evidence_levels': trials['evidence_level'][:],
            'choices': trials['choice'][:],
            'reaction_times': trials['reaction_time'][:]
        }
        
        # Extract neuron information
        neuron_info = {
            'neuron_ids': units.id[:],
            'brain_area': units['brain_area'][:] if 'brain_area' in units.colnames else None,
            'electrode_info': {}
        }
    
    return {
        'spike_times': spike_times,
        'trial_info': trial_info,
        'neuron_info': neuron_info
    }

def create_synthetic_2afc_data(n_neurons: int = 50, 
                              n_trials: int = 200,
                              session_duration: float = 3600.0) -> Dict:
    """
    Create synthetic 2AFC data for testing the pipeline.
    
    Parameters:
    -----------
    n_neurons : int
        Number of neurons to simulate
    n_trials : int  
        Number of trials to simulate
    session_duration : float
        Total session duration in seconds
        
    Returns:
    --------
    dict: Synthetic data in the expected format
    """
    
    print(f"Creating synthetic 2AFC data:")
    print(f"  Neurons: {n_neurons}")
    print(f"  Trials: {n_trials}")
    print(f"  Duration: {session_duration} seconds")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate trial timing
    inter_trial_interval = session_duration / n_trials
    cue_times = np.linspace(10, session_duration - 10, n_trials)
    
    # Add jitter to cue times
    cue_times += np.random.normal(0, 0.5, n_trials)
    
    # Generate variable waiting periods (0.5 to 3 seconds)
    waiting_periods = np.random.uniform(0.5, 3.0, n_trials)
    decision_times = cue_times + waiting_periods
    outcome_times = decision_times + np.random.uniform(0.1, 0.3, n_trials)
    
    # Generate trial outcomes
    # 60% correct, 25% error, 15% timeout
    trial_outcomes = np.random.choice([1, 0, 2], n_trials, p=[0.6, 0.25, 0.15])
    
    # Generate evidence levels (e.g., coherence levels)
    evidence_levels = np.random.choice([0.1, 0.2, 0.4, 0.8], n_trials)
    
    # Generate choices (0=left, 1=right)
    choices = np.random.choice([0, 1], n_trials)
    
    # Calculate reaction times
    reaction_times = decision_times - cue_times
    
    trial_info = {
        'cue_times': cue_times,
        'decision_times': decision_times,
        'outcome_times': outcome_times,
        'trial_outcomes': trial_outcomes,
        'evidence_levels': evidence_levels,
        'choices': choices,
        'reaction_times': reaction_times
    }
    
    # Generate synthetic spike trains
    spike_times = []
    
    for neuron_idx in range(n_neurons):
        # Base firing rate (2-20 Hz)
        base_rate = np.random.uniform(2, 20)
        
        # Generate Poisson spike train
        neuron_spikes = []
        t = 0
        while t < session_duration:
            # Modulate firing rate based on trial phase
            current_rate = base_rate
            
            # Find current trial
            current_trial = np.searchsorted(cue_times, t)
            if current_trial < n_trials:
                # Modulate during waiting period
                if cue_times[current_trial] <= t <= decision_times[current_trial]:
                    # Some neurons increase, others decrease during waiting
                    if neuron_idx % 3 == 0:  # Increase activity
                        current_rate *= (1 + 0.5 * evidence_levels[current_trial])
                    elif neuron_idx % 3 == 1:  # Decrease activity
                        current_rate *= (1 - 0.3 * evidence_levels[current_trial])
                    # Others maintain baseline
                
                # Modulate based on choice
                if t > decision_times[current_trial]:
                    if choices[current_trial] == neuron_idx % 2:  # Preference
                        current_rate *= 1.3
            
            # Generate next spike time
            isi = np.random.exponential(1.0 / current_rate)
            t += isi
            
            if t < session_duration:
                neuron_spikes.append(t)
        
        spike_times.append(np.array(neuron_spikes))
    
    # Create neuron info
    neuron_info = {
        'neuron_ids': np.arange(n_neurons),
        'brain_area': ['simulated'] * n_neurons,
        'electrode_info': {}
    }
    
    return {
        'spike_times': spike_times,
        'trial_info': trial_info,
        'neuron_info': neuron_info
    }

def main():
    """
    Example usage of data loading functions.
    """
    
    print("Example data loading for 2AFC electrophysiology data")
    print("=" * 60)
    
    # Example 1: Create synthetic data
    print("\\n1. Creating synthetic data for testing...")
    synthetic_data = create_synthetic_2afc_data(n_neurons=30, n_trials=150)
    
    print(f"   Created {len(synthetic_data['spike_times'])} neurons")
    print(f"   Created {len(synthetic_data['trial_info']['cue_times'])} trials")
    
    # Save synthetic data for testing
    import pickle
    with open('synthetic_2afc_data.pkl', 'wb') as f:
        pickle.dump(synthetic_data, f)
    print("   Saved synthetic data to 'synthetic_2afc_data.pkl'")
    
    # Example 2: Show how to integrate with the main pipeline
    print("\\n2. Example integration with data preparation pipeline...")
    
    # This is how you would modify the load_spike_data function in prepare_2afc_data.py:
    example_load_function = '''
    def load_spike_data(self, data_path: str) -> Dict:
        """
        Load spike data from your recording format.
        """
        if data_path.endswith('.mat'):
            return load_spike_data_example_matlab(data_path)
        elif data_path.endswith('.pkl'):
            return load_spike_data_example_python(data_path)
        elif data_path.endswith('.nwb'):
            return load_spike_data_example_nwb(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    '''
    
    print("   Modify the load_spike_data function in prepare_2afc_data.py:")
    print(example_load_function)
    
    print("\\n3. Key points for your data:")
    print("   - Ensure spike times are in seconds")
    print("   - Trial outcomes: 0=error, 1=correct, 2=timeout/giveup")
    print("   - All trial timing should be in seconds")
    print("   - Evidence levels can be any numeric scale")
    print("   - Choices: 0/1 or any binary encoding")
    
    print("\\nNext steps:")
    print("1. Modify the load_spike_data function for your data format")
    print("2. Run: python prepare_2afc_data.py --data_path YOUR_DATA.mat --output_path datasets/your_session.h5")
    print("3. Update config files with correct dimensions")
    print("4. Run: python run_2afc_single.py --data_path datasets/your_session.h5 --output_dir results/")
    print("5. Analyze: python analyze_2afc_dynamics.py --model_dir results/ --data_path datasets/your_session.h5 --output_dir analysis/")


if __name__ == "__main__":
    main()