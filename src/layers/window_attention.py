import torch
from torch import nn

class WindowSelfAttention(nn.Module):
    """
    Perform Windowed Self-Attention on a tensor where the batch dimension has each window.
    Note the input and output dimensions don't change, the intuition is that each patch attends to all other patches in the window.
    This updates the representation for each patch, and there are still the same number of patches.
    Window size is denoted as M.

    Parameters
    ----------
    embedding_dim: int
        The dimension of each patch embedding coming into this layer.
        Different transformer blocks might have different values here due to patch merging.
    num_heads: int
        The number of heads to use for self attention.
        The intiution is that in each head, patches attend to other patches differently.
        This is so they can pick out different patterns.
    attention_dropout: float
        The dropout rate to use for the attention weights.
        I.e, it means that randomly some patches won't attend to other patches.
    proj_dropout: float
        The dropout rate to use for the projection that aggregates head information.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dropout_rate: float,
        proj_dropout_rate: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Layers
        self.softmax = nn.Softmax(-1)
        # Store QKV for all heads in a single layer, and decompose later
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(proj_dropout_rate)
    

    def _get_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the windows to QKV and reshape so its [Q,K,V] for each head

        Input: (B x H/M x W/M) x M**2 x C
        Output: 3 x (B x H/M x W/M) x HEADS x M**2 x C/HEADS
        """
        B_, N, C = x.shape()
        qkv = self.qkv(x)
        return qkv.reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
    

    def _self_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Given the tensor container qkv, split it and perform multi-headed self-attention.
        Then concatenate all the heads outputs together.

        Input: 3 x (B x H/M x W/M) x HEADS x M**2 x C/HEADS
        Output: (B x H/M x W/M) x M**2 x C
        """
        THREE, B_, HEADS, N, C_HEADS = qkv.shape
        q,k,v = qkv[0], qkv[1], qkv[2]
        attention_weights = (q @ k.transpose(-2,-1)) * self.head_dim**-0.5
        attention_weights = self.softmax(attention_weights)
        attention_weights = self.attention_dropout(attention_weights)
        # Need to put the head dim next to the embedding dim and flatten it.
        # I.e we need to get the updated representation in each head for each patch, together before we flatten it
        # The projection layer runs on a per patch basis.
        return (attention_weights @ v).tranpose(1,2).view((B_, N, -1))
    

    def _linear_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the concatenated heads, project to the same dimension to mix head information.

        Input: (B x H/M x W/M) x M**2 x C
        Output: (B x H/M x W/M) x M**2 x C
        """
        x = self.proj(x)
        return self.proj_dropout(x)



    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B x H/M x W/M) x M**2 x C
        Output: (B x H/M x W/M) x M**2 x C
        """
        # N === M**2 here
        B_, N, C = x.shape()
        qkv = self._get_qkv(x)
        x = self._self_attention(qkv)
        x = self._linear_projection(x)
        return x


