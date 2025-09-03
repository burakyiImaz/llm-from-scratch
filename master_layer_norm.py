import torch
import torch.nn as nn

class MasterLayerNorm(nn.Module):
    def __init__(self,embedding_dim, eps= 1e-5, device="cpu"):
        super().__init__()
        self.eps= eps
        self.weight= nn.Parameter(torch.ones(embedding_dim,device=device))
        self.device= device

    def forward(self,x):
        mean= x.mean(dim=-1, keepdim=True)
        varience= x.var(dim=-1, keepdim=True, unbiased= False)
        normalized_x= (x-mean) / torch.sqrt(varience+eps)
        return self.weight*normalized_x