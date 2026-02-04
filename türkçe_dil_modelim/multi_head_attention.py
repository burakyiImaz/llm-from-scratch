from causal_attention import CausalAttention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, output_size, num_heads, context_length, dropout_rate=0):
        super().__init__()
       
        self.head_dim = output_size // num_heads
        self.num_heads = num_heads