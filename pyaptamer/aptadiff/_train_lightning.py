"""Lightning wrapper for AptaDiff diffusion training."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffLightning"]

import lightning as L


class AptaDiffLightning(L.LightningModule):
    def __init__(self, diffusion, lr=1e-4, betas=(0.9, 0.999), gamma=0.99):
        super().__init__()
        pass

    def _step(self, batch, stage):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
