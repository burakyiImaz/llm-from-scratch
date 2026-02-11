import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention   
from .layer_norm import LayerNorm
from .mlp import MLP
from .decoder_block import DecoderBlock
from .embedding import Embedding



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

        self.embedding = Embedding(vocab_size, embedding_dim,context_length,device=device)
        self.device = device

        self.layers = nn.Sequential(
    *[
        DecoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            device=device,
            dropout=0.1
        )
        for _ in range(num_layers)
    ]
    )

        self.lm_head = nn.Linear(embedding_dim, vocab_size,device=device)

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

        x = self.embedding(x)


        x = self.layers(x)

        x = self.lm_head(x)

        return x

    def generate(self, x, max_new_tokens, stop_token="."):


        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, seq_len)

        tokens = x[0].tolist()

        # stop token id (örn: ".")
        stop_token_id = "."
        if hasattr(self, "tokenizer"):
            stop_token_id = self.tokenizer.vocab.get(stop_token, None)

        for _ in range(max_new_tokens):

            if len(tokens) > 32:
                break

            logits = self.forward(
                torch.tensor([tokens], device=self.device)
            )

            probs = torch.softmax(logits[0, -1], dim=-1)
            sample= torch.multinomial(probs,1)

            #_, max_index = torch.max(probs, dim=-1)
            max_index= sample.item()
            #next_token = max_index.item()

            tokens.append(max_index)

            if stop_token_id is not None and max_index == stop_token_id:
                break

        return tokens
