import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .config import Config
from .model import Model

class CasualLM(PreTrainedModel):
    config_class= Config

    def __init__(self,config):
        super().__init__(config)
        self.model= Model(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            context_length=config.context_length,
            num_layers=config.context_length,
            device="cpu"
        )
        
        self.lm_head= nn.Linear(config.embedding_dim, config.vocab_size,bias=False)

        self.post_init() # HF weight için önemli

    def forward(self, input_ids):
        return self.model(input_ids)