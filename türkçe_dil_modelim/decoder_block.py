import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .mlp import MLP

class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        device="cpu",
        dropout=0.1
    ):
        super().__init__()

        self.device = device

        self.attention = MultiHeadAttention(
            embed_size=embedding_dim,
            output_size=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout,
            device=device
        )

        self.layer_norm1 = LayerNorm(embedding_dim,eps=1e-5, device="cpu")
        self.layer_norm2 = LayerNorm(embedding_dim,eps=1e-5 ,device="cpu")

        self.mlp = MLP(
            embedding_dim,
            embedding_dim * 4,
            device
        )

    def forward(self, x):
        res = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = x + res


        res = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + res

        return x
