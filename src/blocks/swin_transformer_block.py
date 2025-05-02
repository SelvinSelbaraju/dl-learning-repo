import torch
from torch import nn
from src.layers.utils import WindowSplitter, WindowJoiner, MLP
from src.layers.window_attention import WindowSelfAttention

class SwinTransformerBlock(nn.Module):
    """
    One entire SwinTransformerBlock.
    Has the following steps:
    1. Layer Norm (norm1)
    2. Split into Windows
    3. Windowed Self-Attention
    4. Split back to individual embeddings
    5. Residual connection of steps 1 to 4 with input
    6. Layer Norm (norm2)
    7. MLP layer (with activations)  

    Input: B x (H x W) x C
    Output: B x (H x W) x C
    """
    def __init__(
        self,
        input_resolution: int,
        window_size: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        norm: nn.Module = nn.LayerNorm,
        attention_drop_rate: float = 0.2,
        proj_drop_rate: float = 0.2,
        activation: nn.Module = nn.GELU
    ):
        super().__init__()
        # Metadata
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        # Layers
        self.norm1 = norm(embedding_dim)
        self.norm2 = norm(embedding_dim)
        self.window_splitter = WindowSplitter(
            input_resolution,
            embedding_dim,
            window_size 
        )
        self.window_attention = WindowSelfAttention(
            img_size=input_resolution,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attention_dropout_rate=attention_drop_rate,
            proj_dropout_rate=proj_drop_rate,
            window_size=window_size
        )
        self.window_joiner = WindowJoiner(
            input_resolution=input_resolution,
            embedding_dim=embedding_dim,
            window_size=window_size
        )
        self.mlp = MLP(embedding_dim, hidden_dim, activation)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.window_splitter(x)
        x = self.window_attention(x)
        x = self.window_joiner(x)
        x = x.view((-1, self.input_resolution, self.input_resolution, self.embedding_dim))
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
