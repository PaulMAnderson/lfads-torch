import logging
import os
import warnings
import platform
from glob import glob
from pathlib import Path

import hydra
import torch
import lightning as pl
from hydra.utils import call, instantiate
from omegaconf import OmegaConf, open_dict
from ray import tune

from .utils import flatten

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path(__file__).parent / ".." / p)
)
OmegaConf.register_new_resolver("max", lambda *args: max(args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


def run_model(
    
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline with automatic device detection.
    Supports Apple Silicon (MPS), CUDA GPUs, and CPU.
    """


    # Determine the best available device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Apple Silicon GPU available
        device = "mps"
        # For PyTorch Lightning, we need to use auto accelerator with device specification
        accelerator = "auto"  # Let PyTorch Lightning detect the best accelerator
        devices = 1
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        # NVIDIA GPU available
        device = "cuda"
        accelerator = "gpu"
        devices = torch.cuda.device_count() 
        print(f"Using CUDA GPU with {devices} device(s)")
    else:
        # Fall back to CPU
        device = "cpu"
        accelerator = "cpu"
        devices = None
        print("No GPU detected, using CPU")
    
    # Compose the train config with properly formatted overrides
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(
        config_path=str(config_path.parent),
        job_name="run_model",
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)
    
    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")
    
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)
    
    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)
    if device == "mps":
        model = model.to(device)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in the directory
    ckpt_path = None
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_paths = glob(ckpt_pattern)
        if ckpt_paths:
            ckpt_path = max(ckpt_paths, key=os.path.getctime)
    
    if do_train:
        # If both ray.tune and wandb are being used, ensure that loggers use same name
        try:
            if "single" not in str(config_path) and "wandb_logger" in config.logger:
                with open_dict(config):
                    config.logger.wandb_logger.name = tune.get_trial_name()
                    config.logger.wandb_logger.id = tune.get_trial_name()
        except (ImportError, NameError):
            pass
        
        # if you're on Apple Silicon
        if device == "mps":
            # For newer PyTorch Lightning versions
            trainer = instantiate(
                config.trainer,
                callbacks=[instantiate(c) for c in config.callbacks.values()],
                logger=[instantiate(lg) for lg in config.logger.values()],
                accelerator="auto",  # Let PyTorch Lightning auto-detect
                devices=1,           # Use 1 device
            )
        else:

            # Instantiate the lightning `Trainer` with the appropriate accelerator
            trainer = instantiate(
                config.trainer,
                callbacks=[instantiate(c) for c in config.callbacks.values()],
                logger=[instantiate(lg) for lg in config.logger.values()],
                accelerator=accelerator,
                devices=devices,
            )
        
        # Temporary workaround for PTL step-resuming bug
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            trainer.fit_loop.epoch_loop._batches_that_stepped = ckpt["global_step"]
        
        # Train the model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if ckpt_path else None,
        )
        
        # Restore the best checkpoint if necessary - otherwise, use last checkpoint
        if config.posterior_sampling.use_best_ckpt:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    else:
        if ckpt_path:
            # If not training, restore model from the checkpoint
            model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    
    # Run the posterior sampling function
    if do_posterior_sample:
        # Move model to the appropriate device before posterior sampling
        model = model.to(device)
        call(config.posterior_sampling.fn, model=model, datamodule=datamodule)