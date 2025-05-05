import torch
from src.layers.utils import PatchMerge
from src.blocks.basic_layer import BasicLayer

def test_basic_layer():
    """
    Test that the basic layer runs and outputs the right shape.
    """
    layer = BasicLayer(
        input_resolution=56,
        embedding_dim=96,
        hidden_dim=4*96,
        window_size=2,
        num_heads=3,
        depth=6,
        downsample_layer=PatchMerge(56,96,2*96)
    )
    # These are patch embeddings, imagine 224x224 image split into 4x4 patches.
    example_tensor = torch.ones((2,56**2,96))
    result = layer(example_tensor)
    assert result.shape == (2,28**2,192)
    assert layer.flops() > 10e5
