import os
import shutil
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator


# Need to manually setup ray instance to ensure gpu 
# if not ray.is_initialized():
#     print("Ray not initialized. Starting new instance with 1 GPU...")
#     ray.init(num_gpus=1, include_dashboard=True,)
#     print("Ray successfully initialized.")
# else:
#     print("Ray already initialized. Available resources:")
#     print(ray.cluster_resources())


try:
    # This will connect to an existing Ray instance if one is available
    # Otherwise, it will start a new one
    ray.init(address='auto', 
             ignore_reinit_error=True,
             runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}
             }
    )
    print("Connected to existing Ray instance")
    
    # Get information about the cluster
    print(ray.cluster_resources())
except ConnectionError:
    print("No existing Ray instance found")
    
    # Optionally start a new one
    ray.init(num_gpus=1)
    print("Started new Ray instance")

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "lfads-torch-example"
DATASET_STR = "nlb_mc_maze"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_examplePBT"
RUN_DIR = Path("runs") / PROJECT_STR / DATASET_STR / RUN_TAG
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
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}

# If directory exists, rename it
if RUN_DIR.exists() and RUN_DIR.is_dir():
    counter = 1
    while True:
        # Create a new name with suffix
        new_name = f"{RUN_DIR}_{'0' if counter < 10 else ''}{counter}"
        new_path = Path(new_name)
        
        # If this new name doesn't exist, rename the directory and break
        if not new_path.exists():
            RUN_DIR.rename(new_path)
            break
        
        counter += 1

    # Create the new empty directory
    RUN_DIR.mkdir(exist_ok=True, parents=True)

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
        config_path="../configs/pbt.yaml",
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
    resources_per_trial=dict(cpu=3, gpu=0.5),
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
    config_path="../configs/pbt.yaml",
    do_train=False,
)
