import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5,device="cpu"):
        super().__init__()
        
        self.weights = nn.Parameter(torch.ones(embedding_dim,device=device))
        self.bias= nn.Parameter(torch.zeros(embedding_dim,device=device))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
         
        normalized_x = (x - mean) / (torch.sqrt(var + self.eps))
        out = self.weights * normalized_x
        return out*self.bias