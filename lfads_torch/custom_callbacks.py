# custom_callbacks.py
import ray.tune.integration.pytorch_lightning as rpl
from lightning.pytorch.callbacks import Callback

class CompatibleTuneReportCheckpointCallback(Callback):
    """A compatibility wrapper for TuneReportCheckpointCallback"""
    
    def __init__(self, metrics, filename, on="validation_end"):
        super().__init__()
        self.tune_callback = rpl.TuneReportCheckpointCallback(
            metrics=metrics,
            filename=filename,
            on=on
        )
    
    def on_validation_end(self, trainer, pl_module):
        self.tune_callback.on_validation_end(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        if hasattr(self.tune_callback, "on_train_end"):
            self.tune_callback.on_train_end(trainer, pl_module)