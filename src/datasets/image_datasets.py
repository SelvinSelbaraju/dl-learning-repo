import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets.imagenette import Imagenette
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def create_transform(local_preview: bool = False) -> v2.Transform:
    """
    Create the standard transforms used for image data.
    Specifically:
    1. Take a random area of the image with random aspect ratio and resize to desired size
    2. Randomly flip the image in the horizontal direction
    3. Convert it to a Tensor
    4. Convert it to torch float32
    5. Normalize the image data
    6. Optionally convert it back to PIL format so can visualize it for debugging
    """
    transforms = [
        v2.RandomResizedCrop(size=(224,224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
    if local_preview:
        transforms.append(v2.ToPILImage())
    return v2.Compose(transforms)


def get_imagenette_dataset(root: str, split: str, local_preview: bool = False) -> Imagenette:
    """
    Get an instance of the torchvision Imagenette dataset given the files are local.
    Requires downloading the files first.
    """
    return Imagenette(
        root=root,
        split=split,
        transform=create_transform(local_preview)
    )
