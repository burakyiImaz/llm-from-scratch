import torch
import torch.nn as nn

def get_rotary_position_encoding(input: torch.Tensor, base=10000, device= "cpu"):
    batch_size, context_length, dimension= input.shape

    assert dimension %2 ==0 , "boyut çift olmalı"

    half_dimension= dimension// 2
    freqs_indices= torch.arange(0, half_dimension, device=device, dtype= torch.float32)
    freqs = 1.0 / (base** (freqs_indices/ dimension))
    positions= torch.arange(0, context_length, device=device, dtype= torch.float32)
    angles= positions * freqs
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)      

    input_even = input[:, :, :half_dimension]   # [0,2,4,6,..]
    input_odd= input[:,:,half_dimension:]      # [1,3,5,..]

    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated = input_even * sin_angles + input_odd * cos_angles

    input_rotated= torch.cat([input_even_rotated, input_odd_rotated], dim=-1)
    return input_rotated

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim, context_length, device):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size, embedding_dim, device=device)
        self.get_pos= get_rotary_position_encoding
        self.device= device

    def forward(self,x):
        x= self.embedding(x)
        x= self.get_pos(x,device=self.device)
        return x

