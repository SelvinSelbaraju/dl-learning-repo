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
        patch_size: int = 4,
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
        # We want B x (HW/patch_size^2) x embedding_dim so have 1 embedding row per patch
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

        
class PatchMerge(nn.Module):
    """
    Layer that merges patch embeddings and projects them to a new dimension.
    This layer merges 4 patches into one.
    """
    def __init__(
        self,
        input_resolution: int,
        input_patch_dim: int,
        projection_dim: int,
        use_bias: bool = False,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.input_patch_dim = input_patch_dim
        self.projection_dim = projection_dim
        self.use_bias = use_bias
        # Multiply 4 as 4 patches
        self.proj = nn.Linear(input_patch_dim * 4, projection_dim, bias=use_bias)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects B x (H*W) x C
        Returns B x (H*W/4) x 2C
        """
        B, NUM_PATCHES, CHANNELS = x.shape
        # Check that x has the right shape
        assert (len(x.shape) == 3 and NUM_PATCHES == self.input_resolution**2 and CHANNELS == self.input_patch_dim), f"Expected {B} x {NUM_PATCHES} x {CHANNELS}, got {x.shape}"

        # View as a 4D tensor, so can append together
        # Eg. x0 has the 1st patch, then the 5th patch
        # x1 has the 2nd patch, the 6th patch etc
        # Concat in the last dimension puts the 1st patch in the same row as the second patch etc
        x = x.view(B, self.input_resolution, self.input_resolution, CHANNELS)
        x0 = x[:, 0::2, 0::2, :] # B x H/2 x W/2 x C
        x1 = x[:, 1::2, 0::2, :] # B x H/2 x W/2 x C
        x2 = x[:, 0::2, 1::2, :] # B x H/2 x W/2 x C
        x3 = x[:, 1::2, 1::2, :] # B x H/2 x W/2 x C

        # Combine the patches together and reduce back to 3D
        x = torch.cat([x0, x1, x2, x3], -1) # B x H/2 x W/2 x 4C
        x = x.view((B, -1, 4*self.input_patch_dim)) # B x (H/2 * W/2) x 4C

        return self.proj(x) # B x (H/2 * W/2) x 2C


    def flops(self) -> int:
        # We have self.input_resolution**2 patches, each with a dimension of C
        # We end up with (self.input_resolution**2) / 4 patches, each with a dimension of 2C
        # So like a neural net with input dimension 4C, output dimension 2C, reduced patches
        num_merged_patches = self.input_resolution **2 / 4
        # If we are using the bias, then there is an additional input
        neural_net_flops = (4*self.input_patch_dim + int(self.use_bias)) * self.projection_dim
        return num_merged_patches * neural_net_flops


class WindowSplitter(nn.Module):
    """
    Split a batch of embeddings as a 2d representation into the per window embeddings on a 2D basis

    B x H x W C -> (B x num_windows) x window_size x window_size x C
    """
    def __init__(self, input_resolution: int, embedding_dim: int, window_size: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.embedding_dim = embedding_dim
        self.window_size = window_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        assert H == W == self.input_resolution
        assert C == self.embedding_dim

        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0,1,3,2,4,5).reshape(-1, self.window_size, self.window_size, C)
        return windows

    
