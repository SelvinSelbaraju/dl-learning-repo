from typing import Any
from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    """
    Config to create an Optimizer in Lightning.

    name: str
        The case-identical name of the Optimizer class in torch.optim.
    lr: float
        The learning rate to use.
    kwargs: dict[str, Any]
        Kwargs specific to the optimizer.
    """
    name: str
    lr: float
    kwargs: dict[str, Any] = {}


class TrainingConfig(BaseModel):
    """
    Config for the training loop, apart from the optimizer.

    train_batch_size: int
    val_batch_size: int
    shuffle_train: bool
        Whether the shuffle the data each epoch. Defaults to True.
        Never shuffle validation.
    """
    train_batch_size: int
    val_batch_size: int
    shuffle_train: bool = True
    max_epochs: int = 2
    log_every_n_steps: int = 1



class ArchitectureConfig(BaseModel):
    """
    Simple class for configuring a model.

    name: str
        The name of the model class to use, based on the dict in ModelFactory.
    kwargs: dict[str, Any]
        Model specific kwargs.
    """
    name: str
    kwargs: dict[str, Any]


class ConfigSnapshot(BaseModel):
    optimizer: OptimizerConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
