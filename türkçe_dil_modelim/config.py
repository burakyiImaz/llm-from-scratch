from transformers import PretrainedConfig

class Config(PretrainedConfig):
    model_type = "turkce_llm"

    def __init__(
            self,
            vocab_size=2900,
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            context_length=128,
            **kwargs
    ):
         super().__init__(**kwargs)

         self.vocab_size= vocab_size
         self.embedding_dim= embedding_dim
         self.num_heads= num_heads
         self.num_layers= num_layers
         self.context_length=context_length

         