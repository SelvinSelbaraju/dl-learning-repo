from typing import Tuple
import torch
from torch import nn, optim
import lightning as L

class LightningTrainer(L.LightningModule):
    """
    Lightning wrapper class for ease of use PyTorch training.
    Assumes that the passed model obj has a forward/call method.
    """
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Training step. Assumes that the batch is an input tensor and a label tensor.
        """
        x, labels = batch
        outputs = self.model(x)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
