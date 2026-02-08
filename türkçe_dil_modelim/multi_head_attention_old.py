from .causal_attention import CausalAttention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, output_size, num_heads, context_length, dropout_rate=0):
        super().__init__()
       
        self.head_dim = output_size // num_heads
        self.num_heads = nn.ModuleList([
            CausalAttention(embed_size, self.head_dim, context_length, dropout_rate)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(embed_size, output_size) # çıktıyı istediğimiz formata getirmek için

    def forward(self, x):
        attention_outs= []
        for head in self.num_heads:
            attention_outs.append(head(x))
        attention_outs= torch.cat(attention_outs, dim=-1)
        attention_outs= self.projection(attention_outs)
        return attention_outs