import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, outputs_dim):
        super().__init__()
        self.embed_size = embed_size

        self.q_weights= nn.Linear(embed_size, outputs_dim)
        self.k_weights= nn.Linear(embed_size, outputs_dim)  
        self.v_weights= nn.Linear(embed_size, outputs_dim)

    def forward(self, x):
        q= self.q_weights(x)
        k= self.k_weights(x)
        v= self.v_weights(x)

        attention_scores = q @ k.transpose(-2, -1) # son 2 boyutta değişiklik yapsın diye..
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32)), dim=1)
        return attention_weights @ v

