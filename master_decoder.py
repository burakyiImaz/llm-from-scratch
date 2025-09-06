import torch.nn as nn
from .master_layer_norm import MasterLayerNorm
from .master_mlp import MasterMLP
from .master_multi_head_attention import MasterMultiHeadAttention


class MasterDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, device):
        super().__init__()

        self.self_attention= MasterMultiHeadAttention(
            embedding_dim,
            embedding_dim,
            context_length,
            num_heads,
            dropout_rate=0,
            device=device
        )
        self.norm1= MasterLayerNorm(embedding_dim, device= device)
        self.mlp= MasterMLP(embedding_dim,embedding_dim,device=device)
        self.norm2= MasterLayerNorm(embedding_dim, device=device)

    def forward(x):
        res= self.norm1(x)
        x= self.self_attention(x)
        x= self.norm1(x)

        x= x+res

        res = self.norm2(x)
        x= self.mlp(x)
        x= self.norm2(x)

        x= x+res

        return x