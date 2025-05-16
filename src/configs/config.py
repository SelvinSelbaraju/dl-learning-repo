from typing import Any, Optional, Literal
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
    max_epochs: int
        Stop training after this number of epochs.
        Defaults to 2.
    log_every_n_steps: int
        Log training metrics after this number of steps.
        Defaults to 1.
    checkpoint_monitor: Optional[str]
        Save the best Lightning checkpoint based on this metric.
        Must be one of the metrics logged in the training loop.
        If not provided, best checkpoint saving is disabled.
    load_checkpoint_path: Optional[str]
        Continue training from the checkpoint at the provided path.
        Defaults to None, meaning no checkpoint is used.
    """
    train_batch_size: int
    val_batch_size: int
    shuffle_train: bool = True
    max_epochs: int = 2
    log_every_n_steps: int = 1
    checkpoint_monitor: Optional[str] = None
    checkpoint_mode: Optional[Literal['min', 'max']] = None
    load_checkpoint_path: Optional[str] = None


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
