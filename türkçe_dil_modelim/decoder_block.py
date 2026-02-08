import torch.nn as nn
from .multi_head_attention_old import MultiHeadAttention
from .layer_norm import LayerNorm
from .mlp import MLP


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        self.attention= MultiHeadAttention(embedding_dim, embedding_dim, num_heads, context_length)
        self.layer_norm1= LayerNorm(embedding_dim)
        self.mlp= MLP(embedding_dim, embedding_dim*4)
        self.layer_norm2= LayerNorm(embedding_dim)
    def forward(self, x):
        res= x
        res= self.layer_norm2(res)
        x= self.attention(x)
        x= self.layer_norm1(x)
        x= x + res
        res= x
        res= self.layer_norm2(res)
        x= self.mlp(x)
        x= self.layer_norm2(x)
        x= x + res
        return x
