import pytorch_lightning as pl
from crane.models.base import BaseModel
from typing import Any, Dict
import torch


class LightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for crane.BaseModel.
    
    Handles:
    - Training loop
    - Validation loop
    - Optimizer configuration
    - Logging
    
    Works with ANY model implementing BaseModel interface.
    """
    
    def __init__(
        self,
        model: BaseModel,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, **batch):
        return self.model(**batch)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.model.compute_loss(batch)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        loss = self.model.compute_loss(batch)
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Create optimizer from config
        optimizer_cls = self.optimizer_config.pop('_target_')
        optimizer = optimizer_cls(self.parameters(), **self.optimizer_config)
        
        if self.scheduler_config:
            scheduler_cls = self.scheduler_config.pop('_target_')
            scheduler = scheduler_cls(optimizer, **self.scheduler_config)
            return [optimizer], [scheduler]
        
        return optimizer