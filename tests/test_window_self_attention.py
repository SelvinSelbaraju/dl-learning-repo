import pytest
import torch
from src.layers.window_attention import WindowSelfAttention

# The size we expect after a patch embedding step and window join step
# Patch embed goes from 224 -> 56 in each dim, window has 2x2
B, H, W, M, C = 2,56,56,2,64
NUM_HEADS = 4

@pytest.fixture
def example_layer() -> "WindowSelfAttention":
    return WindowSelfAttention(
        img_size=H,
        embedding_dim=C,
        num_heads=NUM_HEADS,
        attention_dropout_rate=0.0,
        proj_dropout_rate=0.0
    )


def test_project_qkv(example_layer):
    example_tensor = torch.arange(B * H * W * C, dtype=torch.float).view((B*H//M*W//M, M**2, C))
    proj = example_layer._project_qkv(example_tensor)
    assert proj.shape == (B*H//M*W//M, M**2, 3*C)


def test_transform_qkv(example_layer):
    # Each fundamental row is supposed to represent the projection for every patch embedding
    # I.e 3C can be split into 3, where Q, K and V are all length 3
    # Take Q for example, C is then split NUM_HEAD times
    # So with a 64 dimension and 4 heads, the Q projection for each patch has dimension 16 per head
    example_tensor = torch.arange(B * H * W * 3*C, dtype=torch.float).view((B*H//M*W//M, M**2, 3*C))
    res = example_layer._transform_qkv(example_tensor)
    # Check the shape of the result
    assert res.shape == (3, B*H//M*W//M, NUM_HEADS, M**2, C//NUM_HEADS)
    # Let's test Q more precisely
    q = res[0]
    # In order to correctly be calculating attention, we should have M**2 Q projections together for a patch
    # And we have this NUM_HEADS times
    # Let's look in the first window, first head
    q_0_0 = q[0][0]
    # This has the Q projection for each head for all patches in the first window
    # The first head should have the first 4 cols for the 4 rows in the first patch
    # In other words, it should have 0..15 (gap of 3C-16), 3C,3C+1... etc
    expected_0_0 = torch.cat([torch.arange(16).view((1,16)) + (3*C*i) for i in range(4)], dim=0)
    assert q_0_0.shape == expected_0_0.shape
    for r in range(expected_0_0.shape[0]):
        for c in range(expected_0_0.shape[1]):
            assert q_0_0[r][c] == expected_0_0[r][c]


def test_self_attention(example_layer):
    example_tensor = torch.arange(B * H * W * 3*C, dtype=torch.float).view((3, B*H//M*W//M, NUM_HEADS, M**2, C//NUM_HEADS))
    res = example_layer._self_attention(example_tensor)
    assert res.shape == (B*H//M*W//M, M**2, C)


def test_window_self_attention_e2e(example_layer):
    example_tensor = torch.arange(B * H * W * C, dtype=torch.float).view((B*H//M*W//M, M**2, C))
    res = example_layer(example_tensor)
    assert res.shape == (B*H//M*W//M, M**2, C)


# FIXME: Shall I actually test the number of flops? Or just proportionality? 
def test_flops(example_layer):
    print(example_layer.flops())
    

