import torch
from torch import nn
from src.blocks.swin_transformer_block import SwinTransformerBlock

block = SwinTransformerBlock(
    input_resolution=224,
    window_size=4,
    embedding_dim=96,
    num_heads=4,
    hidden_dim=96*4,
)

def test_block():
    example_image = torch.ones((1,224,224,96))
    output = block(example_image)
    # Input shape == output shape
    assert example_image.shape == output.shape
