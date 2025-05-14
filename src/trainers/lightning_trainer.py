from typing import Tuple
import yaml
import torch
from torch import nn, optim
import lightning as L
from torchmetrics.classification import MulticlassAccuracy
from src import logger
from src.configs.config import OptimizerConfig, ConfigSnapshot
from src.models.model_factory import ModelFactory

class LightningTrainer(L.LightningModule):
    """
    Lightning wrapper class for ease of use PyTorch training.
    Assumes that the passed model obj has a forward/call method.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfig,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config

        self.loss = nn.CrossEntropyLoss()
        self.running_train_loss = 0.0
        self.running_train_row_cnt = 0
        self.running_val_loss = 0.0
        self.running_val_row_cnt = 0

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=10, top_k=1)
        self.val_acc = MulticlassAccuracy(num_classes=10, top_k=1)
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Training step. Assumes that the batch is an input tensor and a label tensor.
        """
        x, labels = batch
        outputs = self.model(x)
        loss = self.loss(outputs, labels)
        self.running_train_loss += loss.item()
        self.running_train_row_cnt += x.size(0)
        self.log("batch_train_loss", loss)
        self.log("running_train_loss", self.running_train_loss / self.running_train_row_cnt)
        self.train_acc(outputs, labels)
        self.log("batch_accuracy", self.train_acc, on_step=True, on_epoch=False)
        self.log("running_train_accuracy", self.train_acc.compute(), on_step=True, on_epoch=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log("epoch_train_loss", self.running_train_loss / self.running_train_row_cnt)
        self.running_train_loss = 0.0
        self.running_train_row_cnt = 0


    def validation_step(self, batch):
        x, labels = batch
        outputs = self.model(x)
        loss = self.loss(outputs, labels)
        self.running_val_loss += loss.item()
        self.running_val_row_cnt += x.size(0)
        self.val_acc(outputs, labels)
        self.log("running_validation_accuracy", self.val_acc, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        self.log("epoch_validation_loss", self.running_val_loss / self.running_val_row_cnt)
        self.running_val_loss = 0.0
        self.running_val_row_cnt = 0.0

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer_cls = getattr(optim, self.optimizer_config.name)
        optimizer = optimizer_cls(self.parameters(), self.optimizer_config.lr, **self.optimizer_config.kwargs)
        logger.info(f"Optimizer Details: {optimizer}")
        return optimizer


    @classmethod
    def create_trainer_from_config_path(cls, config_path: str) -> tuple["LightningTrainer",L.Trainer]:
        with open(config_path, "r") as f:
            config = ConfigSnapshot(**yaml.safe_load(f))
        logger.info(f"Creating {config.architecture.name} with config: {config.architecture.kwargs}")
        model = ModelFactory.MODEL_CLASSES[config.architecture.name](**config.architecture.kwargs)
        logger.info(model)
        lightning_module = cls(model, config.optimizer)
        return lightning_module, L.Trainer(
            max_epochs=config.training.max_epochs,
            log_every_n_steps=config.training.log_every_n_steps
        )


