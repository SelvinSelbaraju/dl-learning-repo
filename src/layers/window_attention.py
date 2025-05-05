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
    img_size: int
        The expected H/W of patches this layer should receive.
        For example, a 224x224 image with patch size 4 has 56x56 patches.
        This argument should be 56 in this case.
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
        img_size: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout_rate: float,
        proj_dropout_rate: float,
        window_size: int = 2,
    ):
        super().__init__()
        self.input_resolution = (img_size, img_size)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.window_size = window_size

        # Layers
        self.softmax = nn.Softmax(-1)
        # Store QKV for all heads in a single layer, and decompose later
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(proj_dropout_rate)
    

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the original patch embeddings to 3C.
        Separate method so that can test splitting into QKV works separately.

        Input: (B x H/M x W/M) x M**2 x C
        Output: (B x H/M x W/M) x M**2 x 3C
        """
        return self.qkv(x)

    
    def _transform_qkv(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Given the projection to Q,K,V, reshape and permute.
        This is so we have 3 tensors each with Q,K and V
        Each of these has the NUM_HEADS representations.
        3C represents C for Q, K and V respectively.
        C is split NUM_HEADS times.

        Input: (B x H/M x W/M) x M**2 x 3C
        Output: 3 x (B x H/M x W/M) x HEADS x M**2 x C/HEADS
        """
        B_, N, C_ = qkv.shape
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
        # Cheaper to divide all of Q first, instead of the attention weights
        # Divide less numbers
        q *= self.head_dim**-0.5
        attention_weights = (q @ k.transpose(-2,-1))
        attention_weights = self.softmax(attention_weights)
        attention_weights = self.attention_dropout(attention_weights)
        # Need to put the head dim next to the embedding dim and flatten it.
        # I.e we need to get the updated representation in each head for each patch, together before we flatten it
        # The projection layer runs on a per patch basis.
        return (attention_weights @ v).transpose(1,2).reshape((B_, N, -1))
    

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
        B_, N, C = x.shape
        # Check there is a whole number of batch/image patches
        assert (B_ / (self.input_resolution[0]/self.window_size)**2)
        # N === M**2 here
        assert self.window_size**2 == N
        assert self.embedding_dim == C

        qkv = self._project_qkv(x)
        qkv = self._transform_qkv(qkv)
        x = self._self_attention(qkv)
        x = self._linear_projection(x)
        return x
    

    def flops(self) -> int:
        """
        Calculate the FLOPs for this layer for batch size 1 for a single window.
        The FLOPs are made up of:
        1. Projecting from C to 3C
        2. Dividing Q by a constant 
        3. Multiplying Q and K transpose
        4. Multiping QK_T with V
        5. Projecting from C to C to aggregate head information

        Softmax is ommitted here.
        """
        total = 0
        # Every patch has an embedding (H * W patches)
        # Each of these goes from C with a bias to 3C
        total  += self.window_size**2 * (self.embedding_dim+1) * (self.embedding_dim*3)
        # There are HW/M**2 windows
        # Each window has M**2 projections of dimension C for Q and K 
        # A single window for one head is (M**2 x C//NUM_HEADS) x (C//NUM_HEADS x M**2)
        total += self.num_heads * self.window_size**2 * self.embedding_dim//self.head_dim * self.window_size**2
        # Multiple qkt with v for a single head
        # That is (M**2 x M**2) x (M**2 * C//NUM_HEADS)
        total += self.num_heads * self.window_size**2 * self.window_size**2 * self.embedding_dim//self.head_dim
        # The projection runs for every patch embedding to the same dimension (uses a bias too)
        total += self.window_size**2 * (self.embedding_dim+1) * (self.embedding_dim)
        return total
    
    

