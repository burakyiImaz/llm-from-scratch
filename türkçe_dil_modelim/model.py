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

                                         
    def top_p_filtering(self, logits, top_p):
        sorted_logits, sorted_indices= torch.sort(logits, descending=True)
        cumulative_probs= torch.cumsum(torch.softmax(sorted_logits,dim=-1),dim=-1)
        sorted_indices_to_remove= cumulative_probs>top_p
        sorted_indices_to_remove[...,1:]= sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[...,0]=False

        sorted_logits[sorted_indices_to_remove]= -float('inf')

        filtered_logits= torch.full_like(logits,-float('inf'))
        filtered_logits.scatter_(0,sorted_indices,sorted_logits)

        return filtered_logits

    def generate(self, x, max_new_tokens: int, stop_token=".", temperature: float = 1.0, top_k: int =None, top_p: float=1.0):

        
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
            last_logits = logits[0, -1, :] #sadece son token logitsi
            
            
            if temperature!= 0:
                last_logits= last_logits/temperature
            
            if top_k is not None:
                values, indices = torch.topk(last_logits,top_k)
                filtered_logits= torch.full_like(last_logits,-float("inf"))
                filtered_logits.scatter_(0,indices,values) # önce logitleri sıfırladık sonra top_k ye uygun indice ve değerleri 0 yerine yazdık
                last_logits= filtered_logits
            
            if top_p>0 and top_p<1:
                last_logits= self.top_p_filtering(last_logits,top_p)


            probs = torch.softmax(last_logits, dim=-1)
            

            next_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(next_token)

            if stop_token_id is not None and next_token == stop_token_id:
                break

        return tokens
