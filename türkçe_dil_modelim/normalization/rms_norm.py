import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-8,device="cpu"):

        super().__init__()

        self.weights = nn.Parameter(torch.ones(embedding_dim,device=device))
        self.bias= nn.Parameter(torch.zeros(embedding_dim,device=device))
        self.eps = eps


    def forward(self, x):

        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        x_norm = x / rms

        return x_norm * self.weights

# eğer bias eklersem return x_norm* self.weights + bias yazabilirim ama standartta bias ile kullanılmıyor


