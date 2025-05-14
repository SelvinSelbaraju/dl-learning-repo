import torch
from torch import nn
from src.layers.utils import PatchEmbed, PatchMerge
from src.blocks.basic_layer import BasicLayer

class SwinTransformer(nn.Module):
    """
    Full E2E SwinTransformer model based on https://arxiv.org/abs/2103.14030

    Input: B x C x H x W
    Intermediate Output: B x H_ x W_ x C_
        Where H_ and W_ change due to Patch embedding and Patch merging
        Where C_ changes due to Patch embedding and Patch merging
        The SwinTransformerBlocks keep the dimension constant, they just run Windowed Self-Attention
    Final Output: B x NUM_CLASSES
        Where NUM_CLASSES is dataset specific, for example ImageNet-1k has 1000 classes.
    """
    def __init__(
        self,
        input_resolution: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        embedding_dim: int = 96,
        patch_size: int = 4,
        window_size: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        depths: tuple[int] = (2,2,6,2),
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_layers = len(depths)
        self.num_features = embedding_dim*(2**(self.num_layers-1))
        self.final_resolution = (input_resolution // patch_size) // 2 ** (self.num_layers-1)

        # Layers
        self.patch_embed = PatchEmbed(
            img_size=input_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            # FIXME: Should use_bias ever be False?
        )
        self._layers = [
            BasicLayer(
                # The downsample layer reduces each dimension by a factor of 2 each time
                # So at first divide by 2**0 = 1, then divide by 2, then by 4 etc
                input_resolution=(self.input_resolution//self.patch_size) // 2**i,
                # The downsample layer doubles the embdding dim each time
                embedding_dim=self.embedding_dim*(2**i),
                hidden_dim=self.embedding_dim*mlp_ratio*(2**i),
                window_size=self.window_size,
                num_heads=num_heads,
                depth=depths[i],
                downsample_layer=PatchMerge(
                    input_resolution=(self.input_resolution//self.patch_size) // 2**i,
                    input_patch_dim=self.embedding_dim*(2**i),
                    projection_dim=self.embedding_dim*(2**(i+1))
                    # FIXME: Should I ever not use a bias?
                # Don't downsample in the final layer
                ) if i != self.num_layers-1 else None
            )
            for i in range(self.num_layers)
        ]
        # Required for correct module registration
        self._layers = nn.ModuleList(self._layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            # The output size of the final SwinTransformer layer 
            # If we have 4 SwinTransformer stages, we only double the dimension 3 times
            self.num_features,
            num_classes
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert self.in_channels == C
        assert self.input_resolution == H == W
        # Patch embed
        x = self.patch_embed(x)
        # Pass through all the SwinTransformer + PatchMerge layers
        for layer in self._layers:
            x = layer(x)
        B_INT, N_INT, C_INT = x.shape
        assert B == B_INT
        assert self.final_resolution**2 == N_INT
        assert self.num_features == C_INT
        # Prepare for multi-class classification
        # Need to swap number of patches and channel dimension
        # This is because we aggregate over the final dimension in the avgpool layer
        x = self.avgpool(x.transpose(1,2)) # Outputs B x NUM_FEATURES x 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


    def flops(self) -> int:
        total = 0
        total += self.patch_embed.flops()
        for layer in self._layers:
            total += layer.flops()
        # Avgpool for a batch size of 1
        # We have final_resolution**2 patch embeddings
        # Each of length num_features
        # So for a given dim in num_features, we need to do final_resolution**2 flops
        # Do that for each dim in num_features
        total += (self.num_features * self.final_resolution**2)
        # Head takes the num_features and a bias, and converts that to num_classes
        total += (self.num_features+1) * self.num_classes
        return total


    def num_params(self) -> int:
        # Params come from each the patch embedding + each Swin layer
        # Also from the output head
        total = self.patch_embed.num_params()
        for layer in self._layers:
            total += layer.num_params()
        total += (self.num_features+1)*self.num_classes
        return total


        
        
          
