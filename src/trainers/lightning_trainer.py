from typing import Tuple
import yaml
import torch
from torch import nn, optim
import lightning as L
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
        self.validation_step_outputs = []
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Training step. Assumes that the batch is an input tensor and a label tensor.
        """
        x, labels = batch
        outputs = self.model(x)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch):
        x, labels = batch
        outputs = self.model(x)
        loss = self.loss(outputs, labels)
        top1_class = torch.argmax(outputs, 1)
        equality = torch.eq(top1_class, labels).to(torch.float32)
        self.validation_step_outputs.append({"accuracy": equality, "loss": loss})


    def on_validation_epoch_end(self):
        accuracy = 0
        loss = 0
        length = 0
        for metrics in self.validation_step_outputs:
            accuracy += metrics["accuracy"].sum()
            loss += metrics["loss"].sum()
            length += metrics["accuracy"].shape[0]
        self.log("val_loss", loss / length)
        self.log("val_top1_accuracy", accuracy / length)
        self.validation_step_outputs = []
        

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


