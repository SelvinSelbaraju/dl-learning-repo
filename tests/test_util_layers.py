import torch
import numpy as np
from src.layers.utils import PatchEmbed, PatchMerge, WindowSplitter


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


def _get_constant_patch_embedding(patch_size: int, embedding_dim: int, value: float) -> torch.Tensor:
    """
    Return a patch size tensor with the embedding value fixed.

    Has shape patch_size x patch_size x embedding_dim
    With a fixed value of value everywhere.
    """
    base = torch.ones((patch_size, patch_size, embedding_dim))
    return base * value


def test_window_splitter():
    # Imagine have 8 x 8 patches
    # Made up of 4 (TL, TR, BL, BR) constant patch embeddings
    TL = _get_constant_patch_embedding(4, 5, -1) # 4 x 4 x 5
    TR = _get_constant_patch_embedding(4, 5, 0.5) # 4 x 4 x 5 
    BL = _get_constant_patch_embedding(4, 5, -0.5) # 4 x 4 x 5
    BR = _get_constant_patch_embedding(4, 5, 1) # 4 x 4 x 5
    top = torch.cat([TL, TR], dim=1) # 4 x 8 x 5
    bottom = torch.cat([BL, BR], dim=1) # 4 x 8 x 5
    test_patches = torch.cat([top, bottom], dim=0) # 8 x 8 x 5
    test_batch = torch.stack([test_patches, test_patches]) # 2 x 8 x 8 x 5

    # Should have 8 (2 instances, 4 windows per instance) windows
    # Each window has 4 x 4 elements in, each with a 5 dim embedding
    # So the output of the layer should be of shape 8 x 4 x 4 x 5
    # And the 1st and 5th element should be the same (TL) value as we duplicated the patches
    layer = WindowSplitter(8, 5, 4)
    output = layer(test_batch)
    assert output.shape == (8, 4, 4, 5)
    torch.testing.assert_close(output[0], output[4])

    # Check that the second and sixth elements are TR
    torch.testing.assert_close(output[1], torch.ones((4,4,5))*0.5)
    torch.testing.assert_close(output[5], torch.ones((4,4,5))*0.5)



    

