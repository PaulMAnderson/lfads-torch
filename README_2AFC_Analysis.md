# 2AFC Electrophysiology Data Analysis with LFADS

This repository contains a complete pipeline for analyzing electrophysiology data from 2 Alternative Forced Choice (2AFC) tasks using Latent Factor Analysis via Dynamical Systems (LFADS). The analysis focuses on neural dynamics during waiting periods, comparing correct trials, errors, and trials where animals give up.

## Overview

The pipeline consists of four main components:

1. **Data Preparation** (`prepare_2afc_data.py`): Extracts and formats your electrophysiology data
2. **Configuration** (config files): Sets up LFADS model parameters
3. **Training** (`run_2afc_single.py`): Trains the LFADS model on your data
4. **Analysis** (`analyze_2afc_dynamics.py`): Analyzes learned dynamics and generates visualizations

## Quick Start

### 1. Setup Environment

First, ensure you have the lfads-torch environment set up:

```bash
conda create --name lfads-torch python=3.9
conda activate lfads-torch
cd lfads-torch
pip install -e .
```

### 2. Prepare Your Data

Modify the `load_spike_data` function in `prepare_2afc_data.py` to match your data format. See `example_data_loading.py` for templates for common formats (MATLAB, Python pickle, NWB).

Your data should include:
- **Spike times**: List of arrays, one per neuron
- **Trial information**: Cue times, decision times, outcomes, evidence levels, choices
- **Timing**: All times in seconds

```bash
python prepare_2afc_data.py \\
    --data_path /path/to/your/data.mat \\
    --output_path datasets/session_001.h5 \\
    --bin_width_ms 20 \\
    --waiting_period_ms 2000 \\
    --focus_categories correct error \\
    --plot_summary
```

### 3. Train LFADS Model

```bash
python run_2afc_single.py \\
    --data_path datasets/session_001.h5 \\
    --output_dir results/session_001/ \\
    --max_epochs 200 \\
    --factor_dim 50 \\
    --batch_size 64
```

### 4. Analyze Results

```bash
python analyze_2afc_dynamics.py \\
    --model_dir results/session_001/ \\
    --data_path datasets/session_001.h5 \\
    --output_dir analysis/session_001/
```

## Data Format Requirements

### Input Data Structure

Your data loading function should return a dictionary with:

```python
{
    'spike_times': [array1, array2, ...],  # List of spike time arrays (seconds)
    'trial_info': {
        'cue_times': array,         # Cue onset times (seconds)
        'decision_times': array,    # Decision/response times (seconds)  
        'outcome_times': array,     # Outcome times (seconds)
        'trial_outcomes': array,    # 0=error, 1=correct, 2=timeout/giveup
        'evidence_levels': array,   # Evidence strength (any scale)
        'choices': array,           # 0=left, 1=right (or any binary)
        'reaction_times': array     # Optional: reaction times
    },
    'neuron_info': {
        'neuron_ids': array,        # Neuron identifiers
        'brain_area': list,         # Brain areas (optional)
        'electrode_info': dict      # Additional electrode info (optional)
    }
}
```

### Trial Categorization

Trials are automatically categorized as:
- **Correct**: `trial_outcomes == 1` - Animal made correct choice and was rewarded
- **Error**: `trial_outcomes == 0` - Animal made incorrect choice  
- **Timeout/Giveup**: `trial_outcomes == 2` - Animal abandoned trial before reward

## Configuration Files

### Data Module Configuration (`configs/datamodule/2afc_single_session.yaml`)

Controls data loading and batching:
- `datafile_pattern`: Path to your HDF5 data file
- `batch_size`: Training batch size (adjust for GPU memory)
- `batch_keys`: Additional trial information to load

### Model Configuration (`configs/model/2afc_single_session.yaml`)

Controls LFADS architecture:
- `encod_data_dim`: Number of neurons (auto-detected)
- `encod_seq_len`: Number of time bins (auto-detected)
- `fac_dim`: Latent factor dimensionality (key hyperparameter)
- `gen_dim`: Generator RNN size
- Regularization parameters (`kl_*_scale`, `l2_*_scale`)

## Key Parameters to Tune

### Data Preparation
- `bin_width_ms`: Temporal resolution (10-50ms typical)
- `smoothing_std_ms`: Gaussian smoothing (usually = bin_width_ms)
- `waiting_period_ms`: Duration of waiting period to analyze
- `focus_categories`: Which trial types to include

### Model Architecture
- `fac_dim`: Latent dimensionality (20-100 typical)
- `gen_dim`: Generator size (64-256 typical)
- `co_dim`: Controller outputs (2-8 typical)

### Regularization
- `kl_ic_scale`: Initial condition regularization (1e-7 to 1e-5)
- `kl_co_scale`: Controller output regularization (1e-7 to 1e-5)
- `dropout_rate`: Dropout during training (0.1-0.3)

## Analysis Outputs

The analysis script generates:

### Visualizations
1. **Factor Trajectories**: Dynamics of top latent factors across trial types
2. **Trajectory Distances**: Divergence between trial types over time
3. **PCA Trajectories**: Low-dimensional visualization of waiting period dynamics
4. **Decoding Results**: Classification performance and discriminant weights
5. **Reconstruction Quality**: How well LFADS reconstructs neural activity

### Quantitative Results
- **Decoding Accuracy**: How well trial outcomes can be predicted from factors
- **Reconstruction Correlation**: Quality of neural activity reconstruction
- **Trajectory Distances**: Quantitative separation between trial types
- **PCA Explained Variance**: Dimensionality of neural dynamics

## Expected Results

### Successful Analysis Indicators
1. **High reconstruction correlation** (>0.7): LFADS captures neural activity well
2. **Good decoding accuracy** (>70%): Factors contain trial-relevant information
3. **Clear trajectory separation**: Different trial types follow distinct paths
4. **Smooth factor trajectories**: Dynamics are well-regularized

### Potential Issues
- **Low reconstruction correlation**: Try adjusting regularization, increasing model size
- **Poor decoding**: May need more trials, different factor dimensionality
- **Noisy trajectories**: Increase regularization, check data quality
- **No trial separation**: Verify trial alignment, consider longer time windows

## Multi-Session Analysis

For analyzing multiple sessions from the same or different animals:

1. Prepare each session separately
2. Use multisession LFADS configuration
3. Compute PCR initialization (see `tutorials/multisession/`)
4. Train joint model across sessions

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size, use fewer time bins
2. **Training instability**: Increase regularization, reduce learning rate
3. **Poor convergence**: Try different random seeds, adjust architecture
4. **Data loading errors**: Check data format, verify trial timing

### Performance Tips

1. **GPU usage**: Set `--gpus 1` for faster training
2. **Batch size**: Larger batches (64-256) often work better
3. **Early stopping**: Use patience to avoid overfitting
4. **Hyperparameter search**: Try multiple factor dimensions

## Example Workflow

```bash
# 1. Create synthetic data for testing
python example_data_loading.py

# 2. Prepare the synthetic data
python prepare_2afc_data.py \\
    --data_path synthetic_2afc_data.pkl \\
    --output_path datasets/synthetic_session.h5 \\
    --plot_summary

# 3. Train LFADS model
python run_2afc_single.py \\
    --data_path datasets/synthetic_session.h5 \\
    --output_dir results/synthetic/ \\
    --max_epochs 100 \\
    --factor_dim 20

# 4. Analyze results
python analyze_2afc_dynamics.py \\
    --model_dir results/synthetic/ \\
    --data_path datasets/synthetic_session.h5 \\
    --output_dir analysis/synthetic/
```

## Citation

If you use this pipeline in your research, please cite:

- LFADS-torch: Sedler AR, Pandarinath C. lfads-torch: A modular and extensible implementation of latent factor analysis via dynamical systems. arXiv preprint arXiv:2309.01230. 2023.
- Original LFADS: Pandarinath et al. Inferring single-trial neural population dynamics using sequential auto-encoders. Nature Methods, 15(10):805–815, 2018.

## Support

For questions about:
- **LFADS-torch framework**: See the main repository and Gitter channel
- **2AFC-specific analysis**: Create an issue in this repository
- **Data formatting**: Check `example_data_loading.py` for templates

## File Structure

```
├── prepare_2afc_data.py          # Data preparation script
├── run_2afc_single.py            # Training script  
├── analyze_2afc_dynamics.py      # Analysis script
├── example_data_loading.py       # Data loading examples
├── configs/
│   ├── datamodule/
│   │   └── 2afc_single_session.yaml
│   └── model/
│       └── 2afc_single_session.yaml
├── datasets/                     # Prepared HDF5 files
├── results/                      # Trained models
└── analysis/                     # Analysis outputs
```