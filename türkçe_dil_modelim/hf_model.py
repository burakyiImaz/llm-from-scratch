from transformers import PreTrainedModel
from .config import Config
from .model import Model

class CasualLM(PreTrainedModel):
    config_class= Config

    def __init__(self,config):
        super().__init__(config)

        self.vocab_size= config.vocab_size
        self.embedding_dim= config.embedding_dim
        self.num_heads= config.num_heads
        self.num_layers= config.num_layers
        self.context_length= config.context_length

    def forward(self, input_ids):
        return self.model(input_ids)