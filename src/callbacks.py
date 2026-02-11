from pytorch_lightning.callbacks import Callback, EarlyStopping


class PrintDeviceCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        device = trainer.strategy.root_device
        print(f"Using device: {device}")
