import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from .config import Config
from .model import Model


class CasualLM(PreTrainedModel):
    config_class = Config
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)

        self.model = Model(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            context_length=config.context_length,
            num_layers=config.num_layers,
            device="cpu"
        )

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

        self.post_init()

    def forward(self, input_ids, labels=None):

        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )
