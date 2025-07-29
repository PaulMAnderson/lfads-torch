# 2AFC LFADS Population Based Training (PBT) Setup

This README explains how to use the PBT scripts for optimizing LFADS hyperparameters on 2AFC behavioral task data.

## Quick Start

### 1. Assess Your Data
First, determine whether your dataset is suitable for PBT:

```bash
# python scripts/run_2afc_comparison.py "G:\To Process\PMA17\PMA17 2020-10-22 Session-1\Analysis\LFADS\PMA17_20201022_Session1.h5"
python scripts/run_2afc_comparison.py "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS/PMA17_20201022_Session1.h5"
```

This will analyze your data and recommend whether to use PBT or a single training run.

### 2. Inspect and Prepare Data
Inspect your data file and update the model configuration:

```bash
# python scripts/inspect_2afc_data.py "G:\To Process\PMA17\PMA17 2020-10-22 Session-1\Analysis\LFADS\PMA17_20201022_Session1.h5" --update-config
python scripts/inspect_2afc_data.py "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS/PMA17_20201022_Session1.h5" --update-config
```

### 3. Run PBT (Recommended Approach)
Use the comprehensive preparation and execution script:

```bash
python scripts/prepare_and_run_2afc_pbt.py
```

### 4. Run PBT (Manual Approach)
Alternatively, run PBT directly:

```bash
python scripts/run_2afc_pbt.py
```

## Files Created

The following scripts and configurations have been created for 2AFC PBT:

### Scripts
- `scripts/run_2afc_pbt.py` - Main PBT execution script
- `scripts/prepare_and_run_2afc_pbt.py` - Comprehensive prep and execution
- `scripts/inspect_2afc_data.py` - Data inspection and config updating
- `scripts/run_2afc_comparison.py` - Helps choose between PBT and single run

### Configurations
- `configs/2afc_pbt.yaml` - PBT-specific configuration
- `configs/datamodule/2afc_single_session.yaml` - Data module config (already existed)
- `configs/model/2afc_single_session.yaml` - Model config (already existed)

## PBT Configuration Details

### Hyperparameters Optimized
The PBT script optimizes these key hyperparameters:

1. **Learning Rate** (`model.lr_init`): 1e-5 to 5e-3
2. **Dropout Rate** (`model.dropout_rate`): 0.0 to 0.5
3. **Coordinated Dropout** (`model.train_aug_stack.transforms.0.cd_rate`): 0.05 to 0.4
4. **KL Regularization Scales**:
   - Controller outputs (`model.kl_co_scale`): 1e-7 to 1e-4
   - Initial conditions (`model.kl_ic_scale`): 1e-7 to 1e-4
5. **L2 Regularization Scales**:
   - Generator (`model.l2_gen_scale`): 1e-6 to 1e-2
   - Controller (`model.l2_con_scale`): 1e-6 to 1e-2
6. **Architecture Parameters**:
   - Factor dimensions (`model.fac_dim`): 20 to 80
   - Generator dimensions (`model.gen_dim`): 64 to 256

### PBT Settings
- **Number of trials**: 16 (reduced for single session)
- **Perturbation interval**: 20 epochs
- **Burn-in period**: 80 epochs
- **Early stopping patience**: 6
- **GPU resources**: 0.25 GPU per trial (allows 4 concurrent trials on 1 GPU)

## Customization

### Adjusting PBT Parameters
Edit `scripts/run_2afc_pbt.py` to modify:
- `NUM_TRIALS`: Number of parallel hyperparameter configurations
- `PERTURBATION_INTERVAL`: How often to update hyperparameters
- `BURN_IN_PERIOD`: Wait time before starting perturbations
- `resources_per_trial`: GPU/CPU allocation per trial

### Modifying Hyperparameter Space
In `scripts/run_2afc_pbt.py`, update the `HYPERPARAM_SPACE` dictionary:

```python
HYPERPARAM_SPACE = {
    "model.lr_init": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=3e-3
    ),
    # Add or modify hyperparameters here
}
```

### Changing Data Paths
Update paths in:
- `scripts/run_2afc_pbt.py` (lines with `data_path` and `base_output_dir`)
- `scripts/prepare_and_run_2afc_pbt.py` (in the main function)

## Output Structure

PBT creates the following directory structure:

```
G:\To Process\PMA17\PMA17 2020-10-22 Session-1\Analysis\LFADS\runs\2afc-lfads-pbt\single_session\YYMMDD_2afc_PBT\
├── best_model/                    # Best performing model
│   ├── lightning_checkpoints/     # Model checkpoints
│   ├── csv_logs/                  # Training logs
│   └── lfads_output.h5           # Posterior sampling results
├── 0/                            # Trial 0 results
├── 1/                            # Trial 1 results
├── ...                           # Other trials
├── pbt_config.txt                # PBT configuration
├── best_hyperparameters.txt      # Best hyperparameters found
└── run_2afc_pbt.py              # Copy of execution script
```

## Monitoring Progress

### TensorBoard
Monitor training in real-time:

```bash
tensorboard --logdir "G:\To Process\PMA17\PMA17 2020-10-22 Session-1\Analysis\LFADS\runs\2afc-lfads-pbt"
```

### Weights & Biases (if configured)
If you have W&B set up, you can monitor progress at: https://wandb.ai/your-username/2afc-lfads-pbt

## Troubleshooting

### Common Issues

1. **Out of GPU Memory**
   - Reduce `resources_per_trial` GPU allocation
   - Reduce `NUM_TRIALS` for fewer concurrent trials
   - Reduce model `batch_size` in configs

2. **Data File Not Found**
   - Check that the HDF5 file path is correct
   - Ensure the file contains the expected datasets (`train_encod_data`, etc.)

3. **Ray Initialization Errors**
   - Make sure only one Ray instance is running
   - Restart Python session if Ray gets stuck

4. **Config File Errors**
   - Run `inspect_2afc_data.py` to ensure model config has correct dimensions
   - Check that all config files exist and are properly formatted

### Performance Tips

1. **For small datasets** (< 200 trials, < 20 neurons):
   - Use `run_2afc_single.py` instead of PBT
   - Consider reducing model complexity

2. **For large datasets** (> 1000 trials, > 100 neurons):
   - Increase `NUM_TRIALS` to 32 or 64
   - Extend `BURN_IN_PERIOD` and `PERTURBATION_INTERVAL`
   - Use more aggressive hyperparameter exploration

3. **Limited compute resources**:
   - Reduce `NUM_TRIALS`
   - Increase `resources_per_trial` CPU allocation
   - Run sequentially by setting GPU allocation to 1.0

## Next Steps After PBT

1. **Analyze Results**: The best model will be in `best_model/` directory
2. **Run Posterior Sampling**: Automatically done after PBT completes
3. **2AFC-Specific Analysis**: Use the analysis configuration in `2afc_pbt.yaml`
4. **Compare with Single Run**: Run `run_2afc_single.py` to compare performance

## References

- Original LFADS paper: Pandarinath et al. (2018)
- LFADS-torch documentation: [Add link when available]
- Population Based Training: Jaderberg et al. (2017)