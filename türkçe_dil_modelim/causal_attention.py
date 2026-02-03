import torch
import torch.nn as nn   

class CausalAttention(nn.Module):
    def __init__(self, embed_size, output_size,context_length, dropout_rate=0):
        super().__init__()
        self.embed_size = embed_size
        
        self.q_weights = nn.Linear(embed_size, output_size,bias=False)
        self.k_weights = nn.Linear(embed_size, output_size,bias=False)
        self.v_weights = nn.Linear(embed_size, output_size,bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        context_length= x.shape[0]
        x= x[:context_length]
        Q = self.q_weights(x)  
        K = self.k_weights(x)  
        V = self.v_weights(x)  

        attention_scores= K @ Q.transpose(-2, -1) 
        masked_attention_scores= attention_scores.masked_fill(self.mask==0, -torch.inf)
        attention_weights= torch.softmax(masked_attention_scores/torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32)), dim=-1)
        attention_weights= self.dropout(attention_weights)
        return attention_weights @ V