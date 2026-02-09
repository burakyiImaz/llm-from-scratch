import torch.nn as nn
from .multi_head_attention_old import MultiHeadAttention
from .layer_norm import LayerNorm
from .mlp import MLP


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        device: str = "cpu",
        dropout: float = 0.1
    ):
        super().__init__()

        assert isinstance(dropout, float), f"dropout must be float, got {type(dropout)}"

        self.attention = MultiHeadAttention(
            embedding_dim,
            embedding_dim,
            num_heads,
            context_length,
            device
        )

        self.mlp = MLP(
            embedding_dim,
            embedding_dim * 4,
            device
        )

        self.ln1 = LayerNorm(embedding_dim, device)
        self.ln2 = LayerNorm(embedding_dim, device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ---- Attention block (Pre-LN) ----
        x = x + self.dropout(
            self.attention(self.ln1(x))
        )

        # ---- MLP block (Pre-LN) ----
        x = x + self.dropout(
            self.mlp(self.ln2(x))
        )

        return x
