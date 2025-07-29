import os
import shutil
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator
import h5py

# Need to manually setup ray instance to ensure gpu 
if not ray.is_initialized():
    print("Ray not initialized. Starting new instance with 1 GPU...")
    ray.init(num_gpus=1, include_dashboard=True)
    print("Ray successfully initialized.")
else:
    print("Ray already initialized. Available resources:")
    print(ray.cluster_resources())

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
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
        print(f"  Time bins: {info['n_time_bins']}")
        print(f"  Neurons: {info['n_neurons']}")
        print(f"  Bin width: {info['bin_width_ms']} ms")
        print(f"  Pre-cue bins: {info['pre_cue_bins']}")
        print(f"  Waiting period bins: {info['waiting_period_bins']}")
        print(f"  Post-decision bins: {info['post_decision_bins']}")
        
        return info


# ---------- OPTIONS ----------
data_path = "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS/PMA17_20201022_Session1.h5"
base_output_dir = "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS"

# Training parameters
max_epochs = 200
factor_dim = 50
gen_dim = 128
batch_size = 64

# Create run directory with timestamp
PROJECT_STR = "lfads"
DATASET_STR = "2afc"
RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_pbt"
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

HYPERPARAM_SPACE = {
    "model.lr_init": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=4e-3
    ),
    "model.dropout_rate": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "model.train_aug_stack.transforms.0.cd_rate": HyperParam(
        0.01, 0.7, explore_wt=0.3, enforce_limits=True, init=0.5, sample_fn="uniform"
    ),
    "model.kl_co_scale": HyperParam(1e-6, 1e-4, explore_wt=0.8),
    "model.kl_ic_scale": HyperParam(1e-6, 1e-3, explore_wt=0.8),
    "model.l2_gen_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
    "model.l2_con_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
}
# ------------------------------


# Function to keep dropout and CD rates in-bounds
def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}

# Set the mandatory config overrides to select datamodule and model

mandatory_overrides = {
    "datamodule": "2afc_single_session",  # Use your custom datamodule
    "model": "2afc_single_session",  # Use basic model (or create 2afc_single_session model config)
    # "logger.wandb_logger.project": PROJECT_STR,
    # "logger.wandb_logger.tags.1": DATASET_STR,
    # "logger.wandb_logger.tags.2": RUN_TAG,
    # Model dimensions based on actual data
    "model.encod_data_dim": data_info['n_neurons'],
    "model.encod_seq_len": data_info['n_time_bins'],
    "model.recon_seq_len": data_info['n_time_bins'],
}

# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Run the hyperparameter search
metric = "valid/recon_smth"
num_trials = 20
perturbation_interval = 25
burn_in_period = 80 + 25
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/2afc_pbt.yaml",
        do_posterior_sample=False,
    ),
    metric=metric,
    mode="min",
    name=RUN_DIR.name,
    stop=ImprovementRatioStopper(
        num_trials=num_trials,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=2, gpu=0.3),
    num_samples=num_trials,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
# Copy the best model to a new folder so it is easy to identify
best_model_dir = RUN_DIR / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory to this folder (usually handled by tune)
os.chdir(best_model_dir)
# Load the best model and run posterior sampling (skip training)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint._local_path).name
run_model(
    overrides=mandatory_overrides,
    checkpoint_dir=best_ckpt_dir,
    config_path="../configs/2afc_pbt.yaml",
    do_train=False,
)