import torch
import numpy as np
from src.layers.utils import PatchEmbed


def test_patch_embed():
    # Test a black image
    shape = (1,3,16,16)
    input = torch.zeros(shape)

    layer = PatchEmbed(img_size=16, patch_size=2, in_channels=3, embedding_dim=96, use_bias=False)
    output = layer(input).detach().numpy()
    np.testing.assert_array_equal(
        output,
        torch.zeros((
            1,
            64,
            96
        )).detach().numpy()
    )

