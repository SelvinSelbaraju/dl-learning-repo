import torch
from src.models.swin_transformer import SwinTransformer

def test_swin_transformer():
    """
    Test that the model can have outputs passed through it.
    Check that the output shape is as expected.
    """
    # Use the default settings in the paper for example tensors and model
    example_tensor = torch.ones((2,3,224,224))
    model = SwinTransformer(
        input_resolution=224,
        in_channels=3,
        num_classes=1000,
        embedding_dim=96,
        patch_size=4,
        window_size=7,
        num_heads=3,
        mlp_ratio=4,
        depths=(2,2,6,2)
    )
    res = model(example_tensor)
    assert res.shape == (2,1000)
    # Swin-T has 4.5G FLOPs in the paper, minor differences here (eg. including the bias)
    # Between 1 billion and 10 billion FLOPs
    assert model.flops() > 1e9 and model.flops() < 1e10
