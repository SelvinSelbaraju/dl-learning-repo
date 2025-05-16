from src import logger
from src.datasets.image_datasets import get_imagenette_dataset
from src.datasets.dataloader_factory import DataloaderFactory
from src.trainers.lightning_trainer import LightningTrainer

# FIXME: Make a command line arg
CONFIG_PATH = "/Users/selvino/dl-learning-repo/configs/base_swin.yaml"


train_ds = get_imagenette_dataset(
    root="/Users/selvino/dl-learning-repo/data",
    split="train"
)
val_ds = get_imagenette_dataset(
    root="/Users/selvino/dl-learning-repo/data",
    split="val"
)
train_dataloader, val_dataloader = DataloaderFactory.get_from_config(
    train_ds,
    val_ds,
    CONFIG_PATH
)

lit_module, trainer = LightningTrainer.create_trainer_from_config_path(
    CONFIG_PATH
)
trainer.fit(lit_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=lit_module.load_checkpoint_path)
logger.info(f"Best model checkpoint path: {lit_module.checkpointer.best_model_path}")
