import torch
import torch.nn as nn

"""
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, output_size, num_heads, context_length, dropout_rate=0,device="cpu"):
        super().__init__()
       
        self.context_length= context_length

        self.multi_head_attention= nn.MultiheadAttention(embed_size, num_heads, dropout=dropout_rate,device=device)
        self.projection= nn.Linear(embed_size, output_size,device=device)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        number_of_tokens= x.shape[0]
        x= x[:self.context_length]
        attention_mask= self.mask[:number_of_tokens, :number_of_tokens]
        out, _= self.multi_head_attention(x, x, x, attn_mask= attention_mask) # multi head attention a query key value olarak gönderiyoruz
        #hem maske hali hem de maske olmayan hali var
        out= self.projection(out)
        return out"""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_size,
        output_size,
        num_heads,
        context_length,
        dropout_rate=0.0,
        device="cpu"
    ):
        super().__init__()

        self.context_length = context_length
        self.device = device

        # 🔹 PyTorch MultiheadAttention
        # device parametresi almaz → sonradan .to(device) yapılır
        self.multi_head_attention = nn.MultiheadAttention(
            embed_size,
            num_heads,
            dropout=dropout_rate
        ).to(device)

        # 🔹 Attention çıkışını tekrar embedding boyutuna getirir
        self.projection = nn.Linear(
            embed_size,
            output_size
        ).to(device)

        #  Causal mask (geleceği görmemesi için)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            ).bool()
        )

    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)

        seq_len = x.shape[0]

        #  Context length sınırı
        x = x[:self.context_length]

        #  Maskeyi doğru boyutta al
        attention_mask = self.mask[:seq_len, :seq_len]

        out, _ = self.multi_head_attention(
            x, x, x,
            attn_mask=attention_mask
        )

        out = self.projection(out)
        return out
