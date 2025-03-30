import torch
from torch import nn

class PatchEmbed(nn.Module):
    """
    A layer which maps an image to patch embeddings.

    Args:
    img_size: int
        Excepts square images which have a whole number of patches.
        Num patches = img_size / patch_size^2
    patch_size: int 
        How many pixels to combine.
    in_channels: int
        How many input channels the images have. Usually 3 for RGB.
    embedding_dim: int
        The dimension of patch embeddings.
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int = 2,
        in_channels: int = 3,
        embedding_dim: int = 96,
        use_bias: bool = True,
    ):
        super().__init__()
        self.img_resolution = (img_size, img_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.layer = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, bias=use_bias)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is here to enforce expected shape
        B, C, H, W = x.shape

        # This strictly ensures that all images have the same dimensions
        # This should be relaxed somehow, an investigation for later...
        assert (H,W) == self.img_resolution, f"Images must have shape {self.img_resolution}, got ({H},{W})"

        # Conv2D returns B x embedding_dim x (H/patch_size) x (W / patch_size)
        # We want B x (HW/patch_size) x embedding_dim so have 1 embedding row per patch
        # Therefore we flatten the last 2 dimensions into 1, swap embedding_dim with it.
        return self.layer(x).flatten(2).transpose(1,2)


    def flops(self) -> int:
        """
        Returns the number of flops this layer has.
        """
        num_patches = self.img_resolution[0] * self.img_resolution[0] / self.patch_size**2
        # One patch has patch_size * patch_size ops per in_channel
        # Then we add up all the in_channel outputs
        # Do this for every output channel
        # Do this for every patch
        return (self.patch_size**2 * self.in_channels) * self.embedding_dim * num_patches

        



    
