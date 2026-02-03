import torch
import torch.nn as nn   

class CausalAttention(nn.Module):
    def __init__(self, embed_size, output_size, dropout_rate=0.1):
        super().__init__()
        self.embed_size = embed_size
        
        self.q_weights = nn.Linear(embed_size, output_size,bias=False)
        self.k_weights = nn.Linear(embed_size, output_size,bias=False)
        self.v_weights = nn.Linear(embed_size, output_size,bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.tril(torch.ones(output_size, output_size)))

    def forward(self, x):
        Q = self.q_weights(x)  
        K = self.k_weights(x)  
        V = self.v_weights(x)  

        attention_scores= K @ Q.transpose(-2, -1) / (self.embed_size ** 0.5)
        masked_attention_scores= attention_scores.masked_fill(self.mask==0, float('-inf'))
        attention_weights= torch.softmax(masked_attention_scores, dim=-1)
        attention_weights= self.dropout(attention_weights)
        return attention_weights @ V