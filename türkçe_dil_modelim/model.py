import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention   
from layer_norm import LayerNorm
from mlp import MLP
from decoder_block import DecoderBlock

# kedi köpeği kovaladı , köpek kediyi kovaladı
#yukarıda her ne kadar kelimeler aynı olsa da anlamsal bir farklılık vardır. Bu farklılığı sağlamak için pozisyonel kodlama kullanılır. Deepseek in kullandığı RoPE yaklaşımı

def get_rotary_position_encoding(input, base=10000, device="cpu"):
    batch_size, context_length, dimension = input.shape
    assert dimension % 2 == 0

    half_dim = dimension // 2

    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=device) / dimension))
    positions = torch.arange(0, context_length, device=device)

    angles = positions[:, None] * freqs[None, :]
    sin = torch.sin(angles).unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0)

    x_even = input[:, :, :half_dim]
    x_odd  = input[:, :, half_dim:]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    return torch.cat([x_rot_even, x_rot_odd], dim=-1)

#günümüzde aktif olarak kullanılan bir yöntemdir.
def get_position_encoding(context_length, embedding_dim,base=10000 ,device= "cpu"):
    pos_embedding= torch.zeros(context_length, embedding_dim,device=device)
    for pos in range(context_length):
        for i in range(0, embedding_dim//2):
            pos_embedding[pos, 2*i]= torch.sin(pos / (base ** (2*i/ embedding_dim)))
            if i+1 < embedding_dim//2:
                pos_embedding[pos, 2*i +1]= torch.cos(pos / (base ** (2*i/ embedding_dim)))
    return pos_embedding.unsqueeze(0)  # [1, context_length, embedding_dim] unsqueeze ile batch dimension eklenir ve tensore dönüştürülür.

class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads=2,
        context_length=24,
        device="cpu",
        num_layers=6
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.get_pos = get_rotary_position_encoding
        self.device = device

        self.layers = nn.Sequential(
            *[
                DecoderBlock(
                    embedding_dim,
                    num_heads,
                    context_length
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        # 🔴 ÖNEMLİ: modeli device'a taşı
        self.to(device)

    def forward(self, x):
        # ---- shape güvenliği ----
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # ---- dtype güvenliği ----
        x = x.long().to(self.device)

        # ---- embedding index güvenliği (debug amaçlı) ----
        if torch.any(x >= self.embedding.num_embeddings):
            raise ValueError(
                f"Token id vocab dışı! "
                f"max token={x.max().item()}, "
                f"vocab_size={self.embedding.num_embeddings}"
            )

        # ---- embedding ----
        x = self.embedding(x)

        # ---- rotary position encoding ----
        x = self.get_pos(x, device=self.device)

        # ---- decoder blocks ----
        x = self.layers(x)

        # ---- vocab projection ----
        x = self.lm_head(x)

        return x
