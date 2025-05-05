from typing import Optional
import torch
from torch import nn

from src.layers.utils import PatchMerge
from src.blocks.swin_transformer_block import SwinTransformerBlock

class BasicLayer(nn.Module):
    """
    One "stage" within the SwinTransformer.
    This is composed of a patch downsampling layer (eg. merging)
    Then 1 or more SwinTransformer blocks.

    Input: B x H x W x C
    Output: B x H x W x C if no downsample
        B x H/2 x W/2 x 2C if downsample size 2 on each axis and double channel dim
    """
    def __init__(
        self,
        input_resolution: int,
        embedding_dim: int,
        hidden_dim: int,
        window_size: int,
        num_heads: int,
        depth: int,
        downsample_layer: Optional[PatchMerge] = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.embedding_dim = embedding_dim
        self.blocks = [
            SwinTransformerBlock(
                input_resolution=input_resolution,
                window_size=window_size,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                # FIXME: Set other instance vars here too
                # Left for now as will likely use defaults initially
            )
            for i in range(depth)
        ]
        self.downsample_layer = downsample_layer
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,NUM_PATCHES,C = x.shape
        assert self.input_resolution**2 == NUM_PATCHES
        assert self.embedding_dim == C
        assert self.input_resolution**2 == NUM_PATCHES
        
        for block in self.blocks:
            x = block(x)
        if self.downsample_layer:
            x = self.downsample_layer(x)
        return x


    def flops(self) -> int:
        total = 0
        for block in self.blocks:
            total += block.flops()
        if self.downsample_layer:
            total += self.downsample_layer.flops()
        return total


    def num_params(self) -> int:
        # Params come from TransformerBlocks and PatchMerging layers
        total = 0
        for block in self.blocks:
            total += block.num_params()
        if self.downsample_layer:
            total += self.downsample_layer.num_params()
        return total
        
