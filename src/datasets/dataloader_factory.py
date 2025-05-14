import yaml
from torch.utils.data import Dataset, DataLoader
from src import logger
from src.configs.config import TrainingConfig, ConfigSnapshot

class DataloaderFactory:
    """
    Return a tuple of dataloaders for train and val.
    """
    @staticmethod
    def get_from_config(train_ds: Dataset, val_ds: Dataset, config_path: str) -> tuple[DataLoader, DataLoader]:
        with open(config_path, "r") as f:
            config = ConfigSnapshot(**yaml.safe_load(f))
        training_config = config.training
        logger.info(f"Training config: {training_config}")
        return (
            DataLoader(
                train_ds,
                batch_size=training_config.train_batch_size,
                shuffle=training_config.shuffle_train
            ),
            DataLoader(
                val_ds,
                batch_size=training_config.val_batch_size,
                # Never shuffle validation
                shuffle=False
            )
        )

