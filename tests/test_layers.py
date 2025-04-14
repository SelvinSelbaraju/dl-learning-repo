import torch
import numpy as np
from src.layers.utils import PatchEmbed, PatchMerge


def test_patch_embed():
    # Test a black image
    shape = (1,3,16,16)
    input = torch.zeros(shape)

    layer = PatchEmbed(img_size=16, patch_size=4, in_channels=3, embedding_dim=96, use_bias=False)
    output = layer(input).detach().numpy()
    np.testing.assert_array_equal(
        output,
        torch.zeros((
            1,
            16,
            96
        )).detach().numpy()
    )


def test_patch_merge():
    # Test a black image
    shape = (1,3,16,16)
    input = torch.zeros(shape)

    embed_layer = PatchEmbed(img_size=16, patch_size=4, in_channels=3, embedding_dim=96, use_bias=False)
    embed = embed_layer(input).detach()

    merge_layer = PatchMerge(4, 96, 192)
    result = merge_layer(embed).detach().numpy()
    assert result.shape == (1, 4, 192)

