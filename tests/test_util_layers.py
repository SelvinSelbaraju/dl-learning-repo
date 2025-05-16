import torch
import numpy as np
from src.layers.utils import PatchEmbed, PatchMerge, WindowSplitter, WindowJoiner, MLP


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
    assert output.shape == (8, 16, 5)
    torch.testing.assert_close(output[0], output[4])

    # Check that the second and sixth elements are TR
    torch.testing.assert_close(output[1], torch.ones((16,5))*0.5)
    torch.testing.assert_close(output[5], torch.ones((16,5))*0.5)


def test_window_joiner():
    # Create a dummy M**2 x C tensor
    # Mocks the output of windowed self attention
    # This is the patch embeddings for each patch in that window
    w_emb = torch.ones((4,5))
    # Create 4 different versions of windowed embeddings
    # Stack them in the row dimension
    test_tensor = torch.stack([w_emb*i for i in range(4)]) # 4 x 4 x 5
    # Then make a batch size of 2
    # Use cat not stack as we don't want to be adding a dimension
    # The batch dimension has each window separately in it
    test_tensor = torch.cat([test_tensor, test_tensor]) # 8 x 4 x 5

    layer = WindowJoiner(4,5,2)
    output = layer(test_tensor)
    assert output.shape == (2,16,5)

    # Make sure that the first element in batch has the right ordering
    # The first two embeddings should be 0s, the second two should be 1s
    # The next two should be 0s, the second two should be 1s
    # The next two should be 2s, the second two should be 3s
    torch.testing.assert_close(output[0][0], output[0][4])
    torch.testing.assert_close(output[0][2], output[0][6])
    assert (output[0][2][0] - output[0][1][0]) == 1.0
    assert output[0][8][0] == 2.0
    assert output[0][10][0] == 3.0


def test_mlp():
    # Batch size 2, with 16 patch embeddings of dim 3
    example_tensor = torch.zeros((2,16,3))
    layer = MLP(3,10)
    output = layer(example_tensor)
    # Check output shape
    assert output.shape == (2,16,3)
    # Check layer
    assert layer.layer1.in_features == 3
    assert layer.layer1.out_features == 10
    assert layer.layer2.in_features == 10
    assert layer.layer2.out_features == 3
